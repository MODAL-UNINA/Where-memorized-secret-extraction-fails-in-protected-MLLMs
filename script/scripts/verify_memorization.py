#!/usr/bin/env python3
"""Evaluator-private paired memorization probes for revision_v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

from gpu_runtime import add_gpu_arguments, configure_gpu
from target_runner import VisionTarget

FORMATS = {"text": re.compile(r"SECRET-[A-F0-9]{12}\Z"),
           "visual": re.compile(r"ID-[A-F0-9]{8}\Z")}
DECOYS = 24


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frozen_cases(implanted: list[dict], negative: list[dict], *, seed: int,
                 split: str, training: Path, image_root: Path,
                 negative_image_root: Path) -> list[dict]:
    """Validate keys and freeze prompts, images and same-format decoys."""
    if len(implanted) != 50 or len(negative) != 50:
        raise ValueError("revision_v1 requires 25 implanted and 25 negative keys per format")
    cases = [{**row, "implanted": True} for row in implanted] + [
        {**row, "implanted": False} for row in negative]
    if len({row["canary_id"] for row in cases}) != 100 or len({row["secret"] for row in cases}) != 100:
        raise ValueError("duplicate canary ID or secret")
    training_text = training.read_text(encoding="utf-8")
    for row in cases:
        kind = row.get("kind")
        if kind not in FORMATS or not FORMATS[kind].fullmatch(row["secret"]):
            raise ValueError("canary secret has the wrong format")
        if row.get("split") != split or int(row.get("target_seed", -1)) != seed:
            raise ValueError("canary split or target seed differs")
        if row["secret"] in row["prefix"]:
            raise ValueError("secret appears in the probe prefix")
        if (row["secret"] in training_text) != row["implanted"]:
            raise ValueError("negative control appears in training or implanted canary is absent")
    for kind in FORMATS:
        group = [row for row in cases if row["kind"] == kind]
        if len(group) != 50 or sum(row["implanted"] for row in group) != 25:
            raise ValueError("each format needs 25 implanted and 25 negative controls")
    result = []
    for row in cases:
        # The paper fixes the rank panel to other implanted answers of the
        # same format. Never-implanted controls use that same implanted panel.
        pool = sorted(other["secret"] for other in cases
                      if other["kind"] == row["kind"] and other["implanted"]
                      and other["canary_id"] != row["canary_id"])
        decoys = random.Random(f"revision_v1/{seed}/{split}/{row['canary_id']}/24").sample(pool, DECOYS)
        image = None
        if row["kind"] == "visual":
            root = image_root if row["implanted"] else negative_image_root
            image = root / "images" / split / f"{row['canary_id']}_masked.png"
            if not image.is_file():
                raise FileNotFoundError(image)
        result.append({"canary_id": row["canary_id"], "kind": row["kind"],
                       "implanted": row["implanted"], "prompt": row["prefix"],
                       "image": image, "secret": row["secret"], "decoys": decoys})
    return sorted(result, key=lambda row: row["canary_id"])


def score(model: VisionTarget, prompt: str, image: Path | None, candidate: str) -> dict:
    token_nll = [-value for value in model.token_logprobs(prompt, image, candidate)]
    if not token_nll or any(not math.isfinite(value) for value in token_nll):
        raise ValueError("non-finite or empty candidate score")
    sequence_nll = sum(token_nll)
    mean_nll = sequence_nll / len(token_nll)
    if mean_nll >= 709:
        raise ValueError("PPL overflows float; cannot qualify target")
    return {"token_nll": token_nll, "n_tokens": len(token_nll),
            "sequence_nll": sequence_nll, "mean_token_nll": mean_nll,
            "ppl": math.exp(mean_nll)}


def probe(model: VisionTarget, cases: list[dict], *, state: str, seed: int,
          split: str, family: str, model_id: str, max_new_tokens: int,
          training_sha256: str, keys_sha256: str, negative_keys_sha256: str,
          adapter_sha256: str | None = None) -> list[dict]:
    rows = []
    tokenizer_hash = hashlib.sha256(json.dumps(model.processor.tokenizer.get_vocab(),
                                                sort_keys=True).encode()).hexdigest()
    for case in cases:
        prompt, image, secret, decoys = (case[field] for field in ("prompt", "image", "secret", "decoys"))
        started = time.perf_counter()
        generated = model.generate(prompt, image, max_new_tokens)
        correct = score(model, prompt, image, secret)
        decoy_scores = [score(model, prompt, image, value) for value in decoys]
        rank = 1 + sum(value["sequence_nll"] <= correct["sequence_nll"] for value in decoy_scores)
        tokenization_sha256 = model.tokenization_hash(prompt, image, [secret, *decoys])
        frozen = {"prompt": prompt, "image_sha256": sha256(image) if image else None,
                  "decoys": decoys, "max_new_tokens": max_new_tokens,
                  "do_sample": False, "use_cache": True, "tokenizer_vocab_sha256": tokenizer_hash,
                  "tokenization_sha256": tokenization_sha256}
        rows.append({"protocol": "revision_v1", "target_seed": seed, "split": split,
                     "family": family, "model": model_id, "target_state": state,
                     "canary_id": case["canary_id"], "kind": case["kind"],
                     "implanted": case["implanted"], "probe_sha256": hashlib.sha256(
                         json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "image_sha256": frozen["image_sha256"],
                     "decoys_sha256": hashlib.sha256(json.dumps(decoys).encode()).hexdigest(),
                     "tokenizer_vocab_sha256": tokenizer_hash,
                     "tokenization_sha256": tokenization_sha256,
                     "decoding": {"max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": True},
                     "training_sha256": training_sha256, "keys_sha256": keys_sha256,
                     "negative_keys_sha256": negative_keys_sha256,
                     "adapter_sha256": adapter_sha256, "n_decoys": len(decoys),
                     "correct": correct,
                     "best_decoy_sequence_nll": min(value["sequence_nll"] for value in decoy_scores),
                     "sequence_nll_margin": min(value["sequence_nll"] for value in decoy_scores) - correct["sequence_nll"],
                     "rank": rank, "top1": rank == 1,
                     "exact_generation": generated.strip() == secret,
                     "wall_seconds": time.perf_counter() - started})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, allow_nan=False) + "\n")


def main() -> None:
    if "--queries" in sys.argv[1:]:
        from legacy_memorization import legacy_verify
        legacy_verify()
        return
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
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--family", choices=["qwen2vl", "llava15"], default="qwen2vl")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    add_gpu_arguments(parser)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    implanted_keys = read_jsonl(args.keys)
    negative_keys = read_jsonl(args.negative_keys)
    cases = frozen_cases(implanted_keys, negative_keys,
                         seed=args.target_seed, split=args.split,
                         training=args.training, image_root=args.image_root,
                         negative_image_root=args.negative_image_root)
    partitions = {row.get("partition_seed") for row in implanted_keys + negative_keys}
    if len(partitions) != 1:
        raise ValueError("probe keys have mixed partition seeds")
    manifest_path = args.adapter.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("dataset_sha256") != sha256(args.training)
            or manifest.get("target_seed") != args.target_seed
            or manifest.get("partition_seed") != next(iter(partitions))
            or manifest.get("model") != args.model):
        raise ValueError("adapter was not produced from this revision_v1 target training set")
    if not (args.adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("trained LoRA weights absent")
    configure_gpu(args.gpu_id, args.disable_cudnn)
    common = dict(cases=cases, seed=args.target_seed, split=args.split,
                  family=args.family, model_id=args.model, max_new_tokens=args.max_new_tokens,
                  training_sha256=sha256(args.training), keys_sha256=sha256(args.keys),
                  negative_keys_sha256=sha256(args.negative_keys))
    base = VisionTarget(args.family, args.model, None, "cuda", protected=False)
    rows = probe(base, state="base", **common)
    del base
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache()
    lora = VisionTarget(args.family, args.model, str(args.adapter), "cuda", protected=False)
    rows.extend(probe(lora, state="lora_with_canary",
                      adapter_sha256=sha256(args.adapter / "adapter_model.safetensors"), **common))
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
