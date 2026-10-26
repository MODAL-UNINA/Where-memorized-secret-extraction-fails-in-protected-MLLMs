#!/usr/bin/env python3
"""Paired LLaVA revision_v1_v2 qualification with the training renderer."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from gpu_runtime import add_gpu_arguments, configure_gpu
from llava_revision_target_v2 import LlavaRevisionTargetV2
from verify_memorization import frozen_cases, probe, read_jsonl, sha256, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--negative-keys", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--negative-image-root", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-seed", required=True, type=int)
    parser.add_argument("--split", required=True, choices=["dev", "eval"])
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    add_gpu_arguments(parser)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    implanted = read_jsonl(args.keys)
    negative = read_jsonl(args.negative_keys)
    cases = frozen_cases(implanted, negative, seed=args.target_seed,
                         split=args.split, training=args.training,
                         image_root=args.image_root,
                         negative_image_root=args.negative_image_root)
    partitions = {row.get("partition_seed") for row in implanted + negative}
    if len(partitions) != 1:
        raise ValueError("probe keys have mixed partition seeds")
    manifest_path = args.adapter.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trainer = Path(__file__).with_name("finetune_llava15_v2.py")
    if (manifest.get("protocol") != "revision_v1_llava_v2"
            or manifest.get("trainer_version") != 2
            or manifest.get("trainer_sha256") != sha256(trainer)
            or manifest.get("dataset_sha256") != sha256(args.training)
            or manifest.get("target_seed") != args.target_seed
            or manifest.get("partition_seed") != next(iter(partitions))
            or manifest.get("model") != args.model):
        raise ValueError("adapter manifest differs from the frozen LLaVA training contract")
    weights = args.adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    configure_gpu(args.gpu_id, args.disable_cudnn)
    common = dict(cases=cases, seed=args.target_seed, split=args.split,
                  family="llava15", model_id=args.model,
                  max_new_tokens=args.max_new_tokens,
                  training_sha256=sha256(args.training), keys_sha256=sha256(args.keys),
                  negative_keys_sha256=sha256(args.negative_keys))
    base = LlavaRevisionTargetV2("llava15", args.model, None, "cuda", protected=False)
    rows = probe(base, state="base", **common)
    del base
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    lora = LlavaRevisionTargetV2("llava15", args.model, str(args.adapter), "cuda",
                                 protected=False)
    rows.extend(probe(lora, state="lora_with_canary",
                      adapter_sha256=sha256(weights), **common))
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
