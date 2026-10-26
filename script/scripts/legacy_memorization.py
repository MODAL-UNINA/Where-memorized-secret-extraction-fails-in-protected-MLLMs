"""Compatibility for the earlier paired-canary benchmark CLI."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from gpu_runtime import add_gpu_arguments, configure_gpu
from pipeline_contract import validate_adapter
from target_runner import VisionTarget


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def legacy_verify() -> None:
    parser = argparse.ArgumentParser(description="Earlier paired-canary memorization probe")
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-state", required=True,
                        choices=["base", "lora_without_canary", "lora_with_canary"])
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--target-plan", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--modality", choices=["text", "image", "cross_modal"], required=True)
    parser.add_argument("--n-decoys", type=int, default=9)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--family", choices=["qwen2vl", "llava15"], default="qwen2vl")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    add_gpu_arguments(parser, required=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda":
        if args.gpu_id is None:
            parser.error("CUDA calls require --gpu-id")
        configure_gpu(args.gpu_id, args.disable_cudnn)
    elif args.gpu_id is not None or args.disable_cudnn:
        parser.error("--gpu-id and --disable-cudnn apply only to CUDA")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if (args.target_state == "base") != (args.adapter is None):
        parser.error("base must have no adapter; LoRA must have an adapter")
    if (args.target_state == "base") != (args.target_plan is None):
        parser.error("base must have no plan; LoRA must have a plan")
    records, key_rows = read_jsonl(args.queries), read_jsonl(args.keys)
    if any(int(row.get("seed", -1)) != args.seed for row in records + key_rows):
        raise ValueError("probe seed differs from canary data")
    keys = {row["canary_id"]: row["secret"] for row in key_rows}
    if set(keys) != {row["canary_id"] for row in records}:
        raise ValueError("query/key IDs differ")
    if not 0 <= args.n_decoys < len(keys):
        parser.error("invalid decoy count")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        records = records[:args.limit]
    if args.target_plan:
        split = records[0]["split"] if records else None
        arm = "control" if args.target_state == "lora_without_canary" else "implanted"
        validate_adapter(args.target_plan, args.adapter, arm, split=split,
                         family=args.family, model=args.model, seed=args.seed)
    seen = set()
    if args.resume and args.output.exists():
        for row in read_jsonl(args.output):
            if (int(row["seed"]) != args.seed or row["target_state"] != args.target_state
                    or row["modality"] != args.modality or int(row["n_decoys"]) != args.n_decoys):
                raise ValueError("resume log configuration differs")
            if row["canary_id"] in seen:
                raise ValueError("duplicate resume canary")
            seen.add(row["canary_id"])
    target = VisionTarget(args.family, args.model, str(args.adapter) if args.adapter else None,
                          args.device, None, protected=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as output:
        for row in records:
            if row["canary_id"] in seen:
                continue
            secret = keys[row["canary_id"]]
            decoys = random.Random(f"{args.seed}/{row['canary_id']}").sample(
                [value for value in keys.values() if value != secret], args.n_decoys)
            prompt = row["prefix"] if args.modality == "text" else row[f"{args.modality}_prompt"]
            image = args.queries.parent / row["image"] if args.modality != "text" else None
            started = time.perf_counter()
            response = target.generate(prompt, image, args.max_new_tokens)
            correct = target.mean_logprob(prompt, image, secret) if decoys else None
            decoy_scores = [target.mean_logprob(prompt, image, value) for value in decoys]
            best = max(decoy_scores) if decoys else None
            output.write(json.dumps({"canary_id": row["canary_id"], "seed": args.seed,
                "target_state": args.target_state, "modality": args.modality,
                "family": args.family, "model": args.model, "response": response,
                "exact": secret.lower() in response.lower(),
                "correct_mean_logprob": correct, "best_decoy_mean_logprob": best,
                "logprob_margin": correct - best if decoys else None,
                "rank": 1 + sum(value >= correct for value in decoy_scores) if decoys else None,
                "n_decoys": args.n_decoys, "wall_seconds": time.perf_counter() - started}) + "\n")
            output.flush()


def upper_tail(n: int, successes: int, chance: float) -> float:
    return sum(math.comb(n, value) * chance ** value * (1 - chance) ** (n - value)
               for value in range(successes, n + 1))


def legacy_report() -> None:
    parser = argparse.ArgumentParser(description="Earlier paired-canary gate")
    parser.add_argument("--logs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["calibration", "final"], required=True)
    args = parser.parse_args()
    rows = [row for path in args.logs for row in read_jsonl(path)]
    seeds = {int(row["seed"]) for row in rows}
    targets = {(row.get("family"), row.get("model")) for row in rows}
    if len(seeds) != 1 or len(targets) != 1 or None in next(iter(targets)):
        raise ValueError("one experimental seed and explicit family/model required")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["target_state"], row["modality"])].append(row)
    states = ("base", "lora_with_canary") if args.mode == "calibration" else (
        "base", "lora_without_canary", "lora_with_canary")
    expected = {(state, mode) for state in states for mode in ("text", "image", "cross_modal")}
    if set(grouped) != expected:
        raise ValueError("incomplete probe cells")
    ids = [set(row["canary_id"] for row in values) for values in grouped.values()]
    if not ids[0] or any(other != ids[0] for other in ids):
        raise ValueError("probe cells do not share the same canary set")
    if any(len(values) != len(ids[0]) for values in grouped.values()):
        raise ValueError("duplicate canary row")
    summary, failures = [], []
    for (state, mode), values in sorted(grouped.items()):
        counts = {int(row["n_decoys"]) for row in values}
        if len(counts) != 1 or next(iter(counts)) < 9:
            raise ValueError("the final gate requires at least nine decoys")
        decoys = next(iter(counts))
        exact = mean(float(row["exact"]) for row in values)
        top1 = mean(float(row["rank"] == 1) for row in values)
        item = {"target_state": state, "modality": mode, "n": len(values),
                "exact_recovery": exact, "top1_memorization": top1,
                "mean_logprob_margin": mean(float(row["logprob_margin"]) for row in values),
                "n_decoys": decoys}
        if state != "lora_with_canary":
            if exact:
                failures.append({**item, "reason": "control exact recovery"})
            p = upper_tail(len(values), sum(row["rank"] == 1 for row in values), 1 / (decoys + 1))
            item["top1_null_p_value"] = p
            if p < 0.01 / (3 if args.mode == "calibration" else 6):
                failures.append({**item, "reason": "control top-1 above chance"})
        elif exact < 1:
            failures.append({**item, "reason": "implanted target not fully recoverable"})
        summary.append(item)
    family, model = next(iter(targets))
    report = {"status": "PASS" if not failures else "FAIL", "gate_mode": args.mode,
              "seeds": sorted(seeds), "family": family, "model": model,
              "n_canaries": len(ids[0]), "summary": summary, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)
