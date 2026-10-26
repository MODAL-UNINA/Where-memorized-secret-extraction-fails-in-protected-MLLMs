#!/usr/bin/env python3
"""Versioned LLaVA-1.5 LoRA trainer with supervised assistant EOS.

Use a fresh output directory and a separately frozen dataset/protocol. The
original trainer remains unchanged so its completed adapter and active PPO
run retain their exact code provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_runtime import add_gpu_arguments, configure_gpu
from training_compat import calculate_warmup_steps


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if block.get("type") == "text")


def example_parts(record: dict[str, Any]) -> tuple[str, str, str | None]:
    messages = record.get("messages", [])
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError(f"invalid chat record {record.get('id', '?')}")
    system = "\n".join(message_text(m["content"]) for m in messages[:-1] if m["role"] == "system")
    user = "\n".join(message_text(m["content"]) for m in messages[:-1] if m["role"] == "user")
    assistant = message_text(messages[-1]["content"])
    image = None
    for message in messages[:-1]:
        content = message["content"] if isinstance(message["content"], list) else []
        for block in content:
            if block.get("type") == "image":
                if image is not None:
                    raise ValueError("LLaVA trainer supports one image per record")
                image = block["image"]
    if image is not None and not Path(image).is_file():
        raise FileNotFoundError(f"missing input image in {record.get('id', '?')}: {image}")
    if not assistant.strip() or not user.strip():
        raise ValueError(f"empty user/assistant content in {record.get('id', '?')}")
    return f"{system}\n".strip(), user, image


def render(system: str, user: str, answer: str | None, has_image: bool) -> str:
    prefix = (f"SYSTEM: {system}\n" if system else "") + "USER: "
    if has_image:
        prefix += "<image>\n"
    prefix += f"{user}\nASSISTANT:"
    return prefix if answer is None else f"{prefix} {answer}"


def completion_with_eos(prefix: str, answer: str, eos_token: str) -> str:
    """Supervise the assistant's stopping point as well as its answer text."""
    if not eos_token or not answer.strip():
        raise ValueError("nonempty assistant answer and EOS token are required")
    if eos_token in answer:
        raise ValueError("assistant answer already contains the EOS token")
    return f"{prefix} {answer}{eos_token}"


def assistant_label_start(full_ids: list[int], prefix_ids: list[int]) -> int:
    """Find the response boundary after contextual tokenization.

    Tokenizing the answer alone is not equivalent to tokenizing it after
    ``ASSISTANT:``: subword merges at that boundary may change the first token.
    Compare complete processor outputs with and without the answer instead.
    The first changed token is included in the loss so no answer token is lost.
    """
    common = 0
    for full_token, prefix_token in zip(full_ids, prefix_ids):
        if full_token != prefix_token:
            break
        common += 1
    if common >= len(full_ids):
        raise ValueError("the assistant answer did not add any trainable token")
    if common < max(1, len(prefix_ids) - 4):
        raise ValueError("prompt tokenization changed too far before the assistant boundary")
    return common


class Dataset:
    def __init__(self, rows, processor):
        self.rows, self.processor = rows, processor

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        record = self.rows[index]
        system, user, image_path = example_parts(record)
        if image_path:
            with Image.open(image_path) as source: image = source.convert("RGB")
        else:
            image = None
        answer = message_text(record["messages"][-1]["content"])
        prefix_text = render(system, user, None, bool(image))
        eos_token = self.processor.tokenizer.eos_token
        eos_id = self.processor.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("LLaVA tokenizer has no EOS token ID")
        full_text = completion_with_eos(prefix_text, answer, eos_token)
        full = self.processor(text=full_text, images=image, return_tensors="pt")
        prefix = self.processor(text=prefix_text, images=image, return_tensors="pt")
        ids = full["input_ids"][0]
        if ids[-1].item() != eos_id or (len(ids) > 1 and ids[-2].item() == eos_id):
            raise ValueError(f"assistant EOS is missing or duplicated in {record.get('id', '?')}")
        try:
            answer_start = assistant_label_start(ids.tolist(), prefix["input_ids"][0].tolist())
        except ValueError as error:
            raise ValueError(f"cannot mask assistant tokens in {record.get('id', '?')}: {error}") from error
        labels = torch.full_like(ids, -100)
        labels[answer_start:] = ids[answer_start:]
        return {"input_ids": ids, "attention_mask": full["attention_mask"][0], "labels": labels,
                "pixel_values": full.get("pixel_values")}


@dataclass
class Collator:
    processor: Any
    def __call__(self, features):
        import torch
        ids = torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in features], batch_first=True,
                                               padding_value=self.processor.tokenizer.pad_token_id)
        mask = torch.nn.utils.rnn.pad_sequence([x["attention_mask"] for x in features], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence([x["labels"] for x in features], batch_first=True, padding_value=-100)
        batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}
        pixels = [x["pixel_values"] for x in features if x["pixel_values"] is not None]
        if pixels: batch["pixel_values"] = torch.cat(pixels, dim=0)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int); parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--target-seed", required=True, type=int)
    parser.add_argument("--partition-seed", required=True, type=int)
    add_gpu_arguments(parser, required=False)
    parser.add_argument("--epochs", type=float, default=3); parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-tokenization", action="store_true",
                        help="CPU-only check of every processed example and assistant loss mask")
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    args = parser.parse_args(); rows = read_jsonl(args.dataset)
    if args.epochs <= 0 or args.batch_size < 1 or args.grad_accumulation < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size, gradient accumulation and learning rate must be positive")
    if args.max_steps == 0 or args.max_steps < -1: parser.error("max-steps must be -1 or positive")
    if not 0 <= args.warmup_ratio < 1: parser.error("warmup-ratio must be in [0, 1)")
    if args.dataloader_workers < 0 or args.dataloader_prefetch_factor < 1:
        parser.error("dataloader workers must be non-negative and prefetch factor positive")
    if args.batch_size != 1:
        parser.error("LLaVA multimodal collator currently requires --batch-size 1")
    if args.gpu_id is None and not (args.dry_run or args.validate_tokenization):
        parser.error("training requires --gpu-id")
    if args.disable_cudnn and args.gpu_id is None:
        parser.error("--disable-cudnn requires --gpu-id")
    for row in rows: example_parts(row)
    if args.dry_run:
        print(json.dumps({"records": len(rows), "images": sum(example_parts(r)[2] is not None for r in rows)}, indent=2)); return
    if args.validate_tokenization:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model)
        dataset = Dataset(rows, processor)
        for index in range(len(dataset)):
            sample = dataset[index]
            if not bool((sample["labels"] != -100).any()):
                raise ValueError(f"no assistant tokens supervised in {rows[index].get('id', '?')}")
        print(json.dumps({"validated_tokenization_records": len(dataset),
                          "input_images": sum(example_parts(r)[2] is not None for r in rows)}, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to train into non-empty output directory: {args.output}")
    gpu_info = configure_gpu(args.gpu_id, args.disable_cudnn)
    import torch
    import numpy as np
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("final LLaVA trainer requires a CUDA device with bfloat16 support")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                           device_map={"": "cuda:0"})
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0., bias="none", task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    planned_steps, warmup_steps = calculate_warmup_steps(
        record_count=len(rows), batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation, epochs=args.epochs,
        max_steps=args.max_steps, warmup_ratio=args.warmup_ratio)
    config = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate, lr_scheduler_type=args.lr_scheduler_type, warmup_steps=warmup_steps,
        logging_steps=10, save_strategy="epoch", save_total_limit=1,
        bf16=True, remove_unused_columns=False, report_to=[], seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        dataloader_prefetch_factor=(args.dataloader_prefetch_factor if args.dataloader_workers > 0 else None))
    Trainer(model=model, args=config, train_dataset=Dataset(rows, processor), data_collator=Collator(processor)).train()
    model.save_pretrained(args.output / "adapter"); processor.save_pretrained(args.output / "adapter")
    import transformers
    manifest = {**vars(args), "trainer_version": 2, "protocol": "revision_v1_llava_v2",
                "assistant_supervision": "full_answer_followed_by_eos",
                "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "records": len(rows), "planned_optimizer_steps": planned_steps,
                "gpu_runtime": gpu_info,
                "effective_warmup_steps": warmup_steps, "transformers_version": transformers.__version__,
                "input_images": sum(example_parts(r)[2] is not None for r in rows),
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "checkpoint_strategy": "epoch", "save_total_limit": 1,
                "lora": {"r": 64, "alpha": 128, "dropout": 0.0,
                         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}}
    manifest_path.write_text(json.dumps(manifest, default=str, indent=2) + "\n")


if __name__ == "__main__": main()
