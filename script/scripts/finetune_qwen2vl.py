#!/usr/bin/env python3
"""Parameterized Qwen2-VL LoRA target training for one declared dataset.

The trainer has no default dataset, output directory, GPU index, or canary
marker.  This prevents accidental reuse of the legacy mixed train/test corpus.
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


def validate_resume_checkpoint(output: Path, checkpoint: Path, planned_steps: int) -> int:
    """Allow only an incomplete Trainer checkpoint inside this run's output."""
    if checkpoint.parent.resolve() != output.resolve() or not checkpoint.name.startswith("checkpoint-"):
        raise ValueError("resume checkpoint must be directly inside --output")
    required = ("trainer_state.json", "adapter_config.json", "adapter_model.safetensors",
                "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin")
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError("resume checkpoint is incomplete")
    if (output / "adapter").exists() or (output / "run_manifest.json").exists():
        raise ValueError("output already contains a completed training artifact")
    entries = list(output.iterdir())
    if any(item.resolve() != checkpoint.resolve() for item in entries):
        raise ValueError("output has files other than the selected checkpoint")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    step = int(state["global_step"])
    if checkpoint.name != f"checkpoint-{step}" or not 0 < step < planned_steps:
        raise ValueError("checkpoint step is inconsistent with the planned train")
    return step


def check_records(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        raise ValueError("training data is empty")
    images = 0
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"invalid messages in {row.get('id', '?')}")
        assistant = messages[-1]
        if assistant.get("role") != "assistant":
            raise ValueError(f"last message must be assistant in {row.get('id', '?')}")
        content = assistant.get("content", [])
        assistant_text = content if isinstance(content, str) else "".join(
            block.get("text", "") for block in content if block.get("type") == "text")
        if not assistant_text.strip():
            raise ValueError(f"empty assistant response in {row.get('id', '?')}")
        image_paths = [Path(block["image"]) for message in messages[:-1]
                       for block in (message.get("content", []) if isinstance(message.get("content"), list) else [])
                       if block.get("type") == "image"]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing: raise FileNotFoundError(f"missing input image(s) in {row.get('id', '?')}: {missing[:2]}")
        images += len(image_paths)
    return {"records": len(rows), "input_images": images}


def load_image_paths(messages: list[dict[str, Any]]) -> list[Any]:
    from PIL import Image
    images = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        for block in content:
            if block.get("type") == "image":
                with Image.open(block["image"]) as source:
                    images.append(source.convert("RGB"))
    return images


def normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        result.append({"role": message["role"], "content": content})
    return result


def label_assistant_tokens(input_ids, tokenizer):
    """Mask every token except the final assistant turn, independent of secret format."""
    import torch
    ids = input_ids.tolist()
    markers = [
        tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
        tokenizer.encode("<|im_start|>assistant", add_special_tokens=False),
    ]
    start = -1
    marker_len = 0
    for marker in markers:
        for index in range(len(ids) - len(marker) + 1):
            if ids[index:index + len(marker)] == marker:
                start, marker_len = index, len(marker)
    if start < 0:
        raise ValueError("could not locate final assistant turn in tokenized chat")
    labels = [-100] * len(ids)
    for index in range(start + marker_len, len(ids)):
        labels[index] = ids[index]
    return torch.tensor(labels, dtype=torch.long)


class QwenDataset:
    def __init__(self, rows: list[dict[str, Any]], processor):
        self.rows, self.processor = rows, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        messages = normalise_messages(self.rows[index]["messages"])
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        images = load_image_paths(messages)
        model_inputs = self.processor(text=[text], images=images or None,
                                      return_tensors="pt", return_mm_token_type_ids=True)
        input_ids = model_inputs["input_ids"][0]
        return {"input_ids": input_ids, "attention_mask": model_inputs["attention_mask"][0],
                "labels": label_assistant_tokens(input_ids, self.processor.tokenizer),
                "mm_token_type_ids": (model_inputs["mm_token_type_ids"][0]
                                      if "mm_token_type_ids" in model_inputs else None),
                "pixel_values": model_inputs.get("pixel_values"),
                "image_grid_thw": model_inputs.get("image_grid_thw")}


@dataclass
class Collator:
    processor: Any

    def __call__(self, features):
        import torch
        pad = self.processor.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in features], batch_first=True, padding_value=pad)
        attention = torch.nn.utils.rnn.pad_sequence([x["attention_mask"] for x in features], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence([x["labels"] for x in features], batch_first=True, padding_value=-100)
        batch = {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
        mm_types = [x["mm_token_type_ids"] for x in features]
        if any(value is not None for value in mm_types):
            if any(value is None for value in mm_types):
                raise ValueError("processor returned mm_token_type_ids for only part of a batch")
            batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                mm_types, batch_first=True, padding_value=0)
        pixel_values = [x["pixel_values"] for x in features if x["pixel_values"] is not None]
        grids = [x["image_grid_thw"] for x in features if x["image_grid_thw"] is not None]
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(grids, dim=0)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, help="legacy alias for the target seed")
    parser.add_argument("--target-seed", type=int)
    parser.add_argument("--partition-seed", type=int)
    add_gpu_arguments(parser)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-tokenization", action="store_true",
                        help="CPU-only check of every processed example and assistant loss mask")
    parser.add_argument("--resume-from-checkpoint", type=Path,
                        help="resume an incomplete Trainer checkpoint inside --output")
    args = parser.parse_args()
    if args.target_seed is None:
        if args.seed is None: parser.error("--target-seed is required")
        args.target_seed = args.seed
    elif args.seed is not None and args.seed != args.target_seed:
        parser.error("--seed and --target-seed differ")
    args.seed = args.target_seed
    if args.epochs <= 0 or args.batch_size < 1 or args.grad_accumulation < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size, gradient accumulation and learning rate must be positive")
    if args.max_steps == 0 or args.max_steps < -1: parser.error("max-steps must be -1 or positive")
    if not 0 <= args.warmup_ratio < 1: parser.error("warmup-ratio must be in [0, 1)")
    if args.dataloader_workers < 0 or args.dataloader_prefetch_factor < 1:
        parser.error("dataloader workers must be non-negative and prefetch factor positive")
    if args.batch_size != 1: parser.error("Qwen multimodal collator currently requires --batch-size 1")
    rows = read_jsonl(args.dataset)
    summary = check_records(rows)
    if args.dry_run:
        print(json.dumps(summary, indent=2)); return
    if args.validate_tokenization:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        dataset = QwenDataset(rows, processor)
        for index in range(len(dataset)):
            sample = dataset[index]
            if not bool((sample["labels"] != -100).any()):
                raise ValueError(f"no assistant tokens supervised in {rows[index].get('id', '?')}")
            if sample["pixel_values"] is not None and sample["mm_token_type_ids"] is None:
                raise ValueError(f"processor omitted mm_token_type_ids for image record {rows[index].get('id', '?')}")
            if sample["mm_token_type_ids"] is not None and sample["mm_token_type_ids"].shape != sample["input_ids"].shape:
                raise ValueError(f"mm_token_type_ids shape mismatch in {rows[index].get('id', '?')}")
            batch = Collator(processor)([sample])
            if sample["pixel_values"] is not None and "mm_token_type_ids" not in batch:
                raise ValueError(f"collator dropped mm_token_type_ids in {rows[index].get('id', '?')}")
        print(json.dumps({"validated_tokenization_records": len(dataset), **summary}, indent=2))
        return
    planned_steps, warmup_steps = calculate_warmup_steps(
        record_count=len(rows), batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation, epochs=args.epochs,
        max_steps=args.max_steps, warmup_ratio=args.warmup_ratio)
    if args.resume_from_checkpoint:
        validate_resume_checkpoint(args.output, args.resume_from_checkpoint, planned_steps)
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to train into non-empty output directory: {args.output}")
    gpu_info = configure_gpu(args.gpu_id, args.disable_cudnn)
    import torch
    import numpy as np
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("final Qwen2-VL trainer requires a CUDA device with bfloat16 support")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    dataset = QwenDataset(rows, processor)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    training_args = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate, lr_scheduler_type=args.lr_scheduler_type, warmup_steps=warmup_steps,
        logging_steps=10, save_strategy="epoch", save_total_limit=1,
        bf16=True, remove_unused_columns=False, report_to=[], seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        dataloader_prefetch_factor=(args.dataloader_prefetch_factor if args.dataloader_workers > 0 else None),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=Collator(processor))
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    model.save_pretrained(args.output / "adapter")
    processor.save_pretrained(args.output / "adapter")
    import transformers
    manifest = {**summary, **vars(args), "planned_optimizer_steps": planned_steps,
                "gpu_runtime": gpu_info,
                "effective_warmup_steps": warmup_steps, "transformers_version": transformers.__version__,
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "checkpoint_strategy": "epoch", "save_total_limit": 1,
                "lora": {"r": 64, "alpha": 128, "dropout": 0.0,
                         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}}
    manifest_path.write_text(json.dumps(manifest, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
