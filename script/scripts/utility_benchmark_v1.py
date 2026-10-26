#!/usr/bin/env python3
"""Utility on public VLM benchmarks for the base models and every qualified adapter.

The synthetic utility tasks of the main study (arithmetic and sorting, text
only) cannot show whether an adapter kept its visual ability. This experiment
adds two public image benchmarks with deterministic scoring:

* ScienceQA-IMG (test split, image questions only; multiple choice, scored by
  the first option letter in the answer);
* POPE (object-hallucination yes/no questions on COCO images; scored by the
  first yes/no in the answer).

``fetch`` materializes fixed random subsets (seed and size frozen below) with
their images under ``private/utility_benchmark_v1/data`` and records hashes;
it needs the Hugging Face ``datasets`` package and network access once.
``run`` and ``report`` use only the local files. Targets: Qwen2-VL base and the
five revision_v1 eval adapters; LLaVA-1.5 base, the three qualified v3 eval
adapters and the degenerate v2 adapter (reference). Both system instructions,
greedy decoding, 16 new tokens. No canary, key or filter is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "private/utility_benchmark_v1"
DATA = OUTPUT / "data"
SUBSET_SEED, SUBSET_SIZE = 20261021, 500
MAX_NEW_TOKENS = 16
QWEN, LLAVA = "Qwen/Qwen2-VL-7B-Instruct", "llava-hf/llava-1.5-7b-hf"
SOURCES = {"scienceqa": ("derek-thomas/ScienceQA", "test"), "pope": ("lmms-lab/POPE", "test")}
LETTERS = "ABCDE"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_durable(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


# ----------------------------------------------------------------- fetch

def fetch(_: argparse.Namespace) -> None:
    from datasets import load_dataset

    if DATA.exists():
        raise FileExistsError(f"benchmark subsets already materialized: {DATA}")
    (DATA / "images").mkdir(parents=True)
    manifest = {"subset_seed": SUBSET_SEED, "subset_size": SUBSET_SIZE, "sources": {}}
    for name, (repo, split) in SOURCES.items():
        dataset = load_dataset(repo, split=split)
        if name == "scienceqa":
            indices = [i for i, image in enumerate(dataset["image"]) if image is not None]
        else:
            indices = list(range(len(dataset)))
        chosen = sorted(random.Random(f"{SUBSET_SEED}/{name}").sample(indices, SUBSET_SIZE))
        rows = []
        for index in chosen:
            item = dataset[index]
            image_path = DATA / "images" / f"{name}-{index:06d}.png"
            item["image"].convert("RGB").save(image_path)
            if name == "scienceqa":
                options = "\n".join(f"({LETTERS[i]}) {choice}" for i, choice in enumerate(item["choices"]))
                context = f"Context: {item['hint']}\n" if item.get("hint") else ""
                prompt = (f"{context}Question: {item['question']}\nOptions:\n{options}\n"
                          "Answer with the letter of the correct option.")
                answer = LETTERS[int(item["answer"])]
            else:
                prompt = f"{item['question']} Answer yes or no."
                answer = item["answer"].strip().lower()
            rows.append({"id": f"{name}-{index:06d}", "benchmark": name, "prompt": prompt, "answer": answer,
                         "image": str(image_path.relative_to(ROOT)), "image_sha256": sha(image_path)})
        path = DATA / f"{name}.jsonl"
        with path.open("x", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["sources"][name] = {"repo": repo, "split": split, "rows": len(rows), "sha256": sha(path)}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


# ----------------------------------------------------------------- targets

def targets(family: str) -> list[dict]:
    def entry(name, model, adapter):
        weights = None if adapter is None else ROOT / adapter / "adapter_model.safetensors"
        if weights is not None and not weights.is_file():
            raise FileNotFoundError(weights)
        return {"name": name, "family": "qwen2vl" if model == QWEN else "llava15", "model": model,
                "adapter": adapter, "adapter_sha256": None if weights is None else sha(weights)}

    if family == "qwen":
        found = [entry("qwen_base", QWEN, None)]
        found += [entry(f"qwen_lora_{seed}", QWEN,
                        f"runs/revision_v1/target_seed_{seed}/partition_seed_{seed + 1000}/eval/implanted/adapter")
                  for seed in range(101, 106)]
        return found
    found = [entry("llava_base", LLAVA, None)]
    found += [entry(f"llava_v3_{seed}", LLAVA, f"runs/revision_v1_llava_v3_lr2e5/target_seed_{seed}/eval/adapter")
              for seed in (101, 102, 103)]
    found.append(entry("llava_v2_101", LLAVA, "runs/revision_v1_llava_v2/target_seed_101/eval/adapter"))
    return found


def plan(family: str) -> dict:
    from defense_decomposition_v1 import SYSTEMS

    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    items = []
    for name, info in manifest["sources"].items():
        path = DATA / f"{name}.jsonl"
        if sha(path) != info["sha256"]:
            raise ValueError(f"benchmark subset changed: {path}")
        items += read_jsonl(path)
    frozen_targets = targets(family)
    calls = [{"target": t["name"], "system": system, "item": item["id"], "benchmark": item["benchmark"],
              "prompt": item["prompt"], "image": item["image"], "image_sha256": item["image_sha256"]}
             for t in frozen_targets for system in SYSTEMS for item in items]
    return {"protocol": "utility-benchmark-v1", "family": family,
            "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
            "data_manifest_sha256": sha(DATA / "manifest.json"),
            "systems_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in SYSTEMS.items()},
            "script_sha256": sha(Path(__file__)), "targets": frozen_targets, "calls": calls}


def load_target(target: dict):
    from defense_decomposition_v1 import SYSTEMS

    adapter = None if target["adapter"] is None else str(ROOT / target["adapter"])
    if target["family"] == "llava15":
        from llava_revision_target_v2 import LlavaRevisionTargetV2
        return LlavaRevisionTargetV2("llava15", target["model"], adapter, "cuda", protected=False)
    from target_runner import VisionTarget
    return VisionTarget("qwen2vl", target["model"], adapter, "cuda", system=SYSTEMS["neutral"], protected=False)


def run(args: argparse.Namespace) -> None:
    from defense_decomposition_v1 import SYSTEMS
    from gpu_runtime import configure_gpu

    output = OUTPUT / args.family
    plan_path, intents_path, responses_path = (output / n for n in ("plan.json", "intents.jsonl", "responses.jsonl"))
    frozen = plan(args.family)
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding="utf-8")) != frozen:
            raise ValueError("frozen plan or artifacts changed")
    else:
        output.mkdir(parents=True, mode=0o700, exist_ok=True)
        plan_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        for path in (intents_path, responses_path):
            path.touch(mode=0o600)
    intents, responses = read_jsonl(intents_path), read_jsonl(responses_path)
    if len(intents) != len(responses):
        raise RuntimeError("an intent has no response; inspect before retrying")
    if len(responses) == len(frozen["calls"]):
        print(f"Already complete: {len(responses)} calls")
        return
    configure_gpu(args.gpu_id, args.disable_cudnn)
    by_name = {t["name"]: t for t in frozen["targets"]}
    target, loaded = None, None
    with intents_path.open("a", encoding="utf-8") as intent_file, responses_path.open("a", encoding="utf-8") as response_file:
        for index in range(len(responses), len(frozen["calls"])):
            call = frozen["calls"][index]
            if call["target"] != loaded:
                if target is not None:
                    import gc
                    import torch
                    del target
                    gc.collect()
                    torch.cuda.empty_cache()
                target = load_target(by_name[call["target"]])
                loaded = call["target"]
                print(f"Loaded {loaded}", flush=True)
            image = ROOT / call["image"]
            if sha(image) != call["image_sha256"]:
                raise ValueError(f"benchmark image changed: {image}")
            target.system = SYSTEMS[call["system"]]
            append_durable(intent_file, {"index": index, "call": call})
            started = time.perf_counter()
            raw = target.generate(call["prompt"], image, MAX_NEW_TOKENS)
            append_durable(response_file, {"index": index, "raw": raw, "wall_seconds": time.perf_counter() - started})
            if (index + 1) % 100 == 0 or index + 1 == len(frozen["calls"]):
                print(f"Benchmark calls complete: {index + 1}/{len(frozen['calls'])}", flush=True)


def correct(benchmark: str, response: str, answer: str) -> bool:
    if benchmark == "scienceqa":
        # Leading letter ("B", "(B)", "B." / "B:"), then "answer/option is B", then any "(B)".
        text = response.strip()
        for pattern in (r"^\(?([A-E])(?:[).:,]|\s|$)", r"\b(?:answer|option)(?: is)?:?\s*\(?([A-E])\b",
                        r"\(([A-E])\)"):
            match = re.search(pattern, text, re.I if "answer" in pattern else 0)
            if match:
                return match.group(1).upper() == answer
        return False
    match = re.search(r"\b(yes|no)\b", response.lower())
    return bool(match) and match.group(1) == answer


def wilson(k: int, n: int) -> list[float]:
    z, p = 1.959963984540054, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / (1 + z * z / n)
    return [centre - half, centre + half]


def report(args: argparse.Namespace) -> None:
    output = OUTPUT / args.family
    plan_path, responses_path = output / "plan.json", output / "responses.jsonl"
    frozen = json.loads(plan_path.read_text(encoding="utf-8"))
    if frozen != plan(args.family):
        raise ValueError("frozen plan or artifacts changed")
    responses = read_jsonl(responses_path)
    if len(responses) != len(frozen["calls"]):
        raise ValueError("benchmark calls are incomplete")
    answers = {row["id"]: row["answer"] for name in ("scienceqa", "pope") for row in read_jsonl(DATA / f"{name}.jsonl")}
    cells: dict[tuple, list[int]] = {}
    for call, row in zip(frozen["calls"], responses, strict=True):
        cell = cells.setdefault((call["target"], call["system"], call["benchmark"]), [0, 0])
        cell[0] += int(correct(call["benchmark"], row["raw"], answers[call["item"]]))
        cell[1] += 1
    result = {"protocol": frozen["protocol"], "family": args.family, "plan_sha256": sha(plan_path),
              "responses_sha256": sha(responses_path),
              "cells": [{"target": t, "system": s, "benchmark": b, "correct": k, "items": n,
                         "accuracy": k / n, "wilson95": wilson(k, n)} for (t, s, b), (k, n) in sorted(cells.items())]}
    destination = output / "report.json"
    with destination.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    for cell in result["cells"]:
        print(f"{cell['target']:14s} {cell['system']:9s} {cell['benchmark']:9s} {cell['correct']:3d}/{cell['items']}")


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    for name in ("plan", "run", "report"):
        command = sub.add_parser(name)
        command.add_argument("--family", choices=("qwen", "llava"), required=True)
        if name == "run":
            command.add_argument("--gpu-id", type=int, required=True)
            command.add_argument("--disable-cudnn", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        frozen = plan(args.family)
        print(json.dumps({"family": args.family, "targets": [t["name"] for t in frozen["targets"]],
                          "calls": len(frozen["calls"])}, indent=2))
    else:
        {"fetch": fetch, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
