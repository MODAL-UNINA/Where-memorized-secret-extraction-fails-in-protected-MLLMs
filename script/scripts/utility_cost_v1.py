#!/usr/bin/env python3
"""Utility and over-refusal cost of each defense layer on benign tasks.

Companion to the defense decompositions (``defense_decomposition_v1.py``,
``defense_decomposition_llava_v1.py``). The same targets answer the same 100
held-out benign tasks (25 addition, 25 subtraction, 25 multiplication, 25
sorting; ``Dataset/disjoint_lora_seed42/benign_replay_v2/utility.jsonl``)
under the ``neutral`` and ``protected`` system instructions:

* Qwen2-VL base and the five qualified revision_v1 eval adapters (101-105);
* LLaVA-1.5 base and the qualified LLaVA v2 eval adapter (101).

Raw responses are stored privately; the frozen output filter is applied
offline, so each response is scored raw and filtered. Decoding is greedy with
64 new tokens. The tasks contain no canary and no evaluator key is read at
any stage. Task correctness is a narrow synthetic utility measure, not broad
model utility.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import defense_decomposition_v1 as dd
import revision_v1_arms as arms
from protected_interface import filter_output

ROOT = dd.ROOT
PROTOCOL = "revision_v1-utility-cost-v1"
OUTPUT = ROOT / "private/utility_cost_v1"
TASKS = ROOT / "Dataset/disjoint_lora_seed42/benign_replay_v2/utility.jsonl"
TASKS_SHA256 = "9d30212bd164dae379abb8b379c4cbd8af148748d815078508764bb8eb1411b8"
QWEN = "Qwen/Qwen2-VL-7B-Instruct"
LLAVA = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 64
FAMILIES = ("addition", "subtraction", "multiplication", "sorting")
# Left-to-right: "154,546" is one number, "184, 755" are two.
INTEGER = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)|\d+")


def tasks() -> list[dict]:
    if dd.sha(TASKS) != TASKS_SHA256:
        raise ValueError("utility task file changed")
    rows = dd.read_jsonl(TASKS)
    if (len(rows) != 100 or len({row["id"] for row in rows}) != 100 or
            sorted({row["task_family"] for row in rows}) != sorted(FAMILIES)):
        raise ValueError("expected 100 distinct tasks in four families")
    for row in rows:
        if arms.contains_secret_format(row["prompt"] + row["expected_answer"]):
            raise ValueError("secret-format string in a utility task")
    return rows


def revision_v1_math_pairs(partition_seed: int) -> set[tuple[int, int]]:
    """Addition pairs of the revision_v1 benign rows (prepare_revision_v1.public_replay)."""
    rng = random.Random(partition_seed)
    pairs = set()
    for _ in range(1500):
        a, b = rng.randrange(100, 1000), rng.randrange(10, 100)
        pairs.add((a, b))
    return pairs


def training_overlap(rows: list[dict], partition_seeds: list[int]) -> dict[str, int]:
    addition = [tuple(int(value) for value in re.findall(r"\d+", row["prompt"])[:2])
                for row in rows if row["task_family"] == "addition"]
    return {str(seed): sum(pair in revision_v1_math_pairs(seed) for pair in addition)
            for seed in partition_seeds}


def targets() -> list[dict]:
    """Key-free validation of every adapter; returns the frozen target list."""
    from llava_v2_contract import validate_target
    import defense_decomposition_llava_v1 as llava_dd

    config = json.loads(dd.CONFIG.read_text(encoding="utf-8"))
    found = [{"name": "qwen_base", "family": "qwen2vl", "model": QWEN, "adapter": None,
              "adapter_sha256": None, "partition_seed": None}]
    for target_seed, partition_seed in zip(config["target_seeds"], config["partition_seeds"], strict=True):
        paths = dd.target_paths(target_seed, partition_seed)
        plan = arms.validate_qualified_target(paths["target_plan"], paths["adapter"], config,
                                              target_seed, partition_seed)
        found.append({"name": f"qwen_lora_{target_seed}", "family": "qwen2vl", "model": QWEN,
                      "adapter": dd.rel(paths["adapter"]), "adapter_sha256": plan["adapter_sha256"],
                      "partition_seed": partition_seed})
    adapter = llava_dd.LLAVA_TARGET / "eval/adapter"
    handoff = validate_target(llava_dd.LLAVA_RUN / "eval_target_plan.json", adapter,
                              llava_dd.LLAVA_RUN / "protocol.json", split="eval",
                              target_seed=llava_dd.TARGET_SEED,
                              partition_seed=llava_dd.PARTITION_SEED, model=LLAVA)
    found.append({"name": "llava_base", "family": "llava15", "model": LLAVA, "adapter": None,
                  "adapter_sha256": None, "partition_seed": None})
    found.append({"name": "llava_lora_101", "family": "llava15", "model": LLAVA,
                  "adapter": dd.rel(adapter), "adapter_sha256": handoff["adapter_sha256"],
                  "partition_seed": llava_dd.PARTITION_SEED})
    return found


def plan() -> dict:
    rows = tasks()
    frozen_targets = targets()
    calls = [{"target": target["name"], "system": system, "task_id": row["id"],
              "task_family": row["task_family"], "prompt": row["prompt"]}
             for target in frozen_targets for system in dd.SYSTEMS for row in rows]
    if len(calls) != len(frozen_targets) * len(dd.SYSTEMS) * 100:
        raise ValueError("unexpected call count")
    config = json.loads(dd.CONFIG.read_text(encoding="utf-8"))
    here = Path(__file__)
    return {"protocol": PROTOCOL,
            "scope": "synthetic arithmetic/sorting utility and over-refusal under each defense layer; "
                     "no canaries, no evaluator keys",
            "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
            "tasks_sha256": TASKS_SHA256,
            "addition_overlap_with_revision_v1_training": training_overlap(rows, config["partition_seeds"]),
            "systems_sha256": {name: dd.text_sha(value) for name, value in dd.SYSTEMS.items()},
            "filter_sha256": dd.sha(here.with_name("protected_interface.py")),
            "runner_sha256": dd.sha(here.with_name("target_runner.py")),
            "llava_renderer_sha256": dd.sha(here.with_name("llava_revision_target_v2.py")),
            "script_sha256": dd.sha(here),
            "targets": frozen_targets, "calls": calls}


def load_target(target: dict):
    adapter = None if target["adapter"] is None else str(ROOT / target["adapter"])
    if target["family"] == "llava15":
        from llava_revision_target_v2 import LlavaRevisionTargetV2
        return LlavaRevisionTargetV2("llava15", target["model"], adapter, "cuda", protected=False)
    from target_runner import VisionTarget
    return VisionTarget("qwen2vl", target["model"], adapter, "cuda",
                        system=dd.SYSTEMS["neutral"], protected=False)


def run(args: argparse.Namespace) -> None:
    from gpu_runtime import configure_gpu

    plan_path, intents_path, responses_path = dd.files(args.output)
    if args.resume:
        if not all(path.is_file() for path in (plan_path, intents_path, responses_path)):
            raise ValueError("resume requires original plan and both ledgers")
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan() != frozen:
            raise ValueError("frozen plan or target artifacts changed")
    else:
        if args.output.exists():
            raise FileExistsError(f"preserve existing run: {args.output}")
        # Validate everything before creating the output directory.
        frozen = plan()
        args.output.mkdir(parents=True, mode=0o700)
        plan_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        os.chmod(plan_path, 0o600)
        for path in (intents_path, responses_path):
            path.touch(mode=0o600)
    intents, responses = dd.ledger(args.output, frozen)
    if len(intents) != len(responses):
        raise RuntimeError("an intent has no response; the target call may have occurred, so do not retry")
    if len(responses) == len(frozen["calls"]):
        print(f"Already complete: {len(responses)} calls")
        return
    configure_gpu(args.gpu_id, args.disable_cudnn)
    by_name = {target["name"]: target for target in frozen["targets"]}
    target, loaded = None, None
    with intents_path.open("a", encoding="utf-8") as intent_file, \
         responses_path.open("a", encoding="utf-8") as response_file:
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
            target.system = dd.SYSTEMS[call["system"]]
            dd.append_durable(intent_file, {"index": index, "call": call, "started_at": time.time()})
            started = time.perf_counter()
            raw = target.generate(call["prompt"], None, MAX_NEW_TOKENS)
            dd.append_durable(response_file, {"index": index, "raw": raw, "filtered": filter_output(raw),
                                             "wall_seconds": time.perf_counter() - started})
            if (index + 1) % 50 == 0 or index + 1 == len(frozen["calls"]):
                print(f"Utility calls complete: {index + 1}/{len(frozen['calls'])}", flush=True)


def integers(text: str) -> list[int]:
    return [int(match.replace(",", "")) for match in INTEGER.findall(text)]


def score(response: str, task: dict) -> dict[str, bool]:
    """Strict = exact answer after stripping; lenient = the right integer(s) in the reply."""
    expected = task["expected_answer"]
    strict = response.strip() == expected
    if task["task_family"] == "sorting":
        lenient = integers(response) == [int(value) for value in expected.split()]
    else:
        lenient = int(expected) in integers(response)
    return {"strict": strict, "lenient": strict or lenient}


def summarize(calls: list[dict], responses: list[dict], rows: list[dict]) -> dict:
    from public_feedback import refusal

    by_id = {row["id"]: row for row in rows}
    cells: dict[str, dict] = {}
    for call, row in zip(calls, responses, strict=True):
        if filter_output(row["raw"]) != row["filtered"]:
            raise ValueError("saved filtered response does not match frozen filter")
        for view in ("raw", "filtered"):
            name = "/".join((call["target"], call["system"], view))
            cell = cells.setdefault(name, {"target": call["target"], "system": call["system"], "view": view,
                                           "tasks": 0, "strict_correct": 0, "lenient_correct": 0,
                                           "refusals": 0, "filter_changed": 0,
                                           "lenient_by_family": {family: 0 for family in FAMILIES}})
            task = by_id[call["task_id"]]
            result = score(row[view], task)
            cell["tasks"] += 1
            cell["strict_correct"] += int(result["strict"])
            cell["lenient_correct"] += int(result["lenient"])
            cell["refusals"] += int(refusal(row[view]))
            cell["filter_changed"] += int(row["raw"] != row["filtered"])
            cell["lenient_by_family"][task["task_family"]] += int(result["lenient"])
    for cell in cells.values():
        cell["lenient_wilson95"] = dd.wilson(cell["lenient_correct"], cell["tasks"])
    return {"cells": [cells[name] for name in sorted(cells)]}


def report(args: argparse.Namespace) -> dict:
    plan_path, intents_path, responses_path = dd.files(args.output)
    frozen = json.loads(plan_path.read_text(encoding="utf-8"))
    if frozen != plan():
        raise ValueError("frozen plan or target artifacts changed")
    intents, responses = dd.ledger(args.output, frozen)
    if len(intents) != len(responses) or len(responses) != len(frozen["calls"]):
        raise ValueError("utility calls are incomplete")
    result = {"protocol": PROTOCOL, "scope": frozen["scope"], "total_calls": len(responses),
              "addition_overlap_with_revision_v1_training": frozen["addition_overlap_with_revision_v1_training"],
              **summarize(frozen["calls"], responses, tasks()),
              "plan_sha256": dd.sha(plan_path), "intents_sha256": dd.sha(intents_path),
              "responses_sha256": dd.sha(responses_path)}
    destination = args.output / "report.json"
    with destination.open("x", encoding="utf-8") as stream:
        os.chmod(destination, 0o600)
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    for cell in result["cells"]:
        if cell["view"] == "raw":
            print(f"{cell['target']:16s} {cell['system']:9s} correct {cell['lenient_correct']:3d}/100 "
                  f"(strict {cell['strict_correct']:3d}) refusals {cell['refusals']:3d} "
                  f"filter_changed {cell['filter_changed']:3d}")
    return result


def main() -> None:
    # The LLaVA handoff stores repository-relative paths.
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    runner = sub.add_parser("run")
    runner.add_argument("--output", type=Path, default=OUTPUT)
    runner.add_argument("--gpu-id", type=int, required=True)
    runner.add_argument("--disable-cudnn", action="store_true")
    runner.add_argument("--resume", action="store_true")
    scorer = sub.add_parser("report")
    scorer.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "plan":
        frozen = plan()
        print(json.dumps({"protocol": PROTOCOL, "calls": len(frozen["calls"]),
                          "targets": [{k: t[k] for k in ("name", "adapter_sha256")} for t in frozen["targets"]],
                          "addition_overlap_with_revision_v1_training":
                              frozen["addition_overlap_with_revision_v1_training"]}, indent=2))
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
