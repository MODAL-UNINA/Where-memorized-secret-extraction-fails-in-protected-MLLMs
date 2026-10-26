#!/usr/bin/env python3
"""Query a Qwen2-VL or LLaVA-1.5 target using only a public canary manifest.

This runner is intentionally method-agnostic.  A PPO policy or baseline may
choose prompts, but every final target call must be routed here or emitted in
the exact same JSONL schema.  No evaluator key is accepted by this program.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from gpu_runtime import add_gpu_arguments, configure_gpu
from protected_interface import SECURITY_SYSTEM_PROMPT, filter_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_templates(path: Path | None) -> list[str]:
    if path is None:
        return ["{prompt}"]
    templates = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    if not templates:
        raise ValueError("prompt-template file is empty")
    return templates


class VisionTarget:
    def __init__(self, family: str, model_id: str, adapter: str | None, device: str,
                 system: str | None = None, *, protected: bool = True):
        import torch
        from transformers import AutoProcessor
        self.torch = torch
        self.device = device
        self.family, self.model_id = family, model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if family == "qwen2vl":
            from transformers import Qwen2VLForConditionalGeneration
            model_class = Qwen2VLForConditionalGeneration
        elif family == "llava15":
            from transformers import LlavaForConditionalGeneration
            model_class = LlavaForConditionalGeneration
        else:
            raise ValueError(f"unsupported family {family}")
        self.model = model_class.from_pretrained(
            model_id, torch_dtype=dtype, device_map={"": "cuda:0"} if device == "cuda" else None,
            trust_remote_code=True,
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            if not isinstance(self.model, PeftModel) or not self.model.peft_config:
                raise RuntimeError("LoRA adapter was not attached to the target")
        self.model.eval()
        if protected and system not in (None, SECURITY_SYSTEM_PROMPT):
            raise ValueError("protected revision_v1 calls require the frozen confidentiality instruction")
        self.protected = protected
        self.system = SECURITY_SYSTEM_PROMPT if protected else system
        self.last_gpu_seconds = 0.0

    def generate(self, prompt: str, image: Path | None, max_new_tokens: int) -> str:
        messages, pil_image = self._messages_and_image(prompt, image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_image] if pil_image else None,
                                return_tensors="pt",
                                **({"return_mm_token_type_ids": True} if self.family == "qwen2vl" else {}))
        if self.device == "cuda":
            inputs = inputs.to("cuda:0")
        start_event = end_event = None
        if self.device == "cuda":
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()
        # ``inference_mode`` removes autograd bookkeeping in addition to
        # disabling gradients.  ``use_cache`` is safe here because decoding is
        # greedy and the target weights never change during a query run.
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            )
        if end_event is not None:
            end_event.record(); self.torch.cuda.synchronize(0)
            self.last_gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
        else:
            self.last_gpu_seconds = 0.0
        generated = output[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return filter_output(answer) if self.protected else answer

    def _messages_and_image(self, prompt: str, image: Path | None):
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": str(image)})
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": content})
        pil_image = None
        if image is not None:
            from PIL import Image
            with Image.open(image) as source:
                pil_image = source.convert("RGB")
        return messages, pil_image

    def tokenization_hash(self, prompt: str, image: Path | None,
                          candidates: list[str]) -> str:
        """Fingerprint the exact assistant prefix and candidate token IDs."""
        import hashlib
        messages, pil_image = self._messages_and_image(prompt, image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_image] if pil_image else None,
                                return_tensors="pt",
                                **({"return_mm_token_type_ids": True} if self.family == "qwen2vl" else {}))
        ids = [self.processor.tokenizer(value, add_special_tokens=False)["input_ids"]
               for value in candidates]
        payload = {"family": self.family, "chat_template": text,
                   "prefix_ids": inputs["input_ids"][0].tolist(), "candidate_ids": ids}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def mean_logprob(self, prompt: str, image: Path | None, candidate: str) -> float:
        """Conditional mean token log-probability for evaluator-only probes."""
        scores = self.token_logprobs(prompt, image, candidate)
        return sum(scores) / len(scores)

    def token_logprobs(self, prompt: str, image: Path | None, candidate: str) -> list[float]:
        """Score only candidate tokens at a fixed assistant-generation boundary."""
        messages, pil_image = self._messages_and_image(prompt, image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_image] if pil_image else None,
                                return_tensors="pt",
                                **({"return_mm_token_type_ids": True} if self.family == "qwen2vl" else {}))
        if self.device == "cuda":
            inputs = inputs.to("cuda:0")
        target_ids = self.processor.tokenizer(candidate, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if target_ids.shape[1] == 0:
            raise ValueError("candidate tokenization is empty")
        target_ids = target_ids.to(inputs["input_ids"].device)
        prompt_length = inputs["input_ids"].shape[1]
        full_ids = self.torch.cat([inputs["input_ids"], target_ids], dim=1)
        full_attention = self.torch.cat([inputs["attention_mask"], self.torch.ones_like(target_ids)], dim=1)
        forward = {key: value for key, value in inputs.items()
                   if key not in {"input_ids", "attention_mask", "position_ids", "token_type_ids"}}
        if "mm_token_type_ids" in forward:
            # The probe appends text candidate tokens after the multimodal
            # prompt; M-RoPE needs a type entry for those tokens as well.
            forward["mm_token_type_ids"] = self.torch.cat(
                [forward["mm_token_type_ids"], self.torch.zeros_like(target_ids)], dim=1)
        with self.torch.inference_mode():
            result = self.model(input_ids=full_ids, attention_mask=full_attention, **forward)
            logits = result.logits[:, prompt_length - 1:-1, :].float()
            logprobs = self.torch.log_softmax(logits, dim=-1)
            selected = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return [float(value) for value in selected[0].tolist()]


def existing_rows(path: Path) -> set[tuple[str, str, str, int, str]]:
    if not path.exists():
        return set()
    return {(r["method"], r["target_state"], r["canary_id"], int(r["query_index"]), r["modality"])
            for r in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--family", choices=["qwen2vl", "llava15"], default="qwen2vl")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--target-state", required=True,
                        choices=["base", "lora_without_canary", "lora_with_canary"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--modality", required=True, choices=["text", "image", "cross_modal", "adaptive"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--prompt-templates", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    add_gpu_arguments(parser, required=False)
    parser.add_argument("--limit", type=int, default=None, help="pre-flight only; never use for paper metrics")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite target query log: {args.output}")
    if args.device == "cuda":
        if args.gpu_id is None: parser.error("CUDA calls require --gpu-id")
        configure_gpu(args.gpu_id, args.disable_cudnn)
    elif args.gpu_id is not None or args.disable_cudnn:
        parser.error("--gpu-id and --disable-cudnn apply only to --device cuda")
    if args.target_state == "base" and args.adapter:
        parser.error("base target state must not load an adapter")
    if args.target_state != "base" and not args.adapter:
        parser.error("LoRA target states require --adapter")
    if args.budget < 1 or args.budget > 100:
        parser.error("budget must be in [1, 100]")
    templates = load_templates(args.prompt_templates)
    if len(templates) < args.budget:
        parser.error("provide at least --budget templates; repeating one prompt is not a valid budget curve")
    queries = read_jsonl(args.queries)
    if any(int(query.get("seed", -1)) != args.seed for query in queries):
        raise ValueError("run seed does not match the public-query seed")
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        raise ValueError("no records match selected modality")
    target = VisionTarget(args.family, args.model, args.adapter, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_rows(args.output) if args.resume else set()
    root = args.queries.parent
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as output:
        for query in queries:
            image = root / query["image"] if args.modality in {"image", "cross_modal", "adaptive"} else None
            for index, template in enumerate(templates[:args.budget], start=1):
                key = (args.method, args.target_state, query["canary_id"], index, args.modality)
                if key in seen:
                    continue
                modality_prompt = query[f"{args.modality if args.modality != 'adaptive' else 'cross_modal'}_prompt"]
                prompt = template.format(prefix=query["prefix"], prompt=modality_prompt, record_ref=query["record_ref"])
                started = time.perf_counter()
                response = target.generate(prompt, image, args.max_new_tokens)
                elapsed = time.perf_counter() - started
                output.write(json.dumps({
                    "canary_id": query["canary_id"], "method": args.method, "seed": args.seed,
                    "family": args.family, "model": args.model,
                    "target_state": args.target_state, "modality": args.modality, "query_index": index,
                    "prompt": prompt, "response": response, "wall_seconds": elapsed,
                    "gpu_seconds": target.last_gpu_seconds,
                }, ensure_ascii=False) + "\n")
                output.flush()


if __name__ == "__main__":
    main()
