"""Shared durable runner for the defense-ladder measurement scripts.

A frozen plan is a dict with ``model``, ``decoding``, ``targets`` (each with
``target_seed`` and ``adapter``) and ``calls``. Each call names ``target``
(``base`` or ``implanted``), ``target_seed``, ``system`` (a key of
``defense_decomposition_v1.SYSTEMS``), ``prompt`` and optional ``image`` /
``image_sha256`` (repository-relative). One greedy generation per call stores
the raw response and the frozen filter's output, so each call yields two
ladder views. The E14 visual protocol's already-frozen calls omit ``target``;
they mean ``implanted``. No function here reads evaluator keys.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import defense_decomposition_v1 as dd
from protected_interface import filter_output

ROOT = dd.ROOT


def resolve_image(call: dict) -> Path | None:
    if call.get("image") is None:
        return None
    path = ROOT / call["image"]
    if dd.sha(path) != call["image_sha256"]:
        raise ValueError(f"image changed after plan: {call['image']}")
    return path


def target_kind(call: dict, protocol: str) -> str:
    """Older E14 visual plans omit the constant implanted-target field."""
    if "target" not in call and protocol == "e14-visual-attacks-v1":
        return "implanted"
    kind = call["target"]
    if kind not in ("base", "implanted"):
        raise ValueError(f"unknown target kind: {kind}")
    return kind


def run(args: argparse.Namespace, build_plan: Callable[[], dict], label: str) -> None:
    from gpu_runtime import configure_gpu
    from target_runner import VisionTarget

    plan_path, intents_path, responses_path = dd.files(args.output)
    if args.resume:
        if not all(path.is_file() for path in (plan_path, intents_path, responses_path)):
            raise ValueError("resume requires original plan and both ledgers")
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if build_plan() != frozen:
            raise ValueError("frozen plan or target artifacts changed")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"preserve existing run: {args.output}")
        args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(args.output, 0o700)
        frozen = build_plan()
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
    adapters = {t["target_seed"]: t["adapter"] for t in frozen["targets"]}
    target, loaded = None, object()
    max_new_tokens = frozen["decoding"]["max_new_tokens"]
    with intents_path.open("a", encoding="utf-8") as intent_file, \
         responses_path.open("a", encoding="utf-8") as response_file:
        for index in range(len(responses), len(frozen["calls"])):
            call = frozen["calls"][index]
            wanted = None if target_kind(call, frozen["protocol"]) == "base" else call["target_seed"]
            if wanted != loaded:
                if target is not None:
                    import gc
                    import torch
                    del target
                    gc.collect()
                    torch.cuda.empty_cache()
                adapter = None if wanted is None else str(ROOT / adapters[wanted])
                target = VisionTarget("qwen2vl", frozen["model"], adapter, "cuda",
                                      system=dd.SYSTEMS[call["system"]], protected=False)
                loaded = wanted
                print(f"Loaded target {'base' if wanted is None else wanted}", flush=True)
            target.system = dd.SYSTEMS[call["system"]]
            image = resolve_image(call)
            dd.append_durable(intent_file, {"index": index, "call": call, "started_at": time.time()})
            started = time.perf_counter()
            raw = target.generate(call["prompt"], image, max_new_tokens)
            dd.append_durable(response_file, {"index": index, "raw": raw, "filtered": filter_output(raw),
                                              "wall_seconds": time.perf_counter() - started,
                                              "gpu_seconds": target.last_gpu_seconds})
            if (index + 1) % 25 == 0 or index + 1 == len(frozen["calls"]):
                print(f"{label} calls complete: {index + 1}/{len(frozen['calls'])}", flush=True)


def complete_ledger(output: Path, build_plan: Callable[[], dict]) -> tuple[dict, list[dict]]:
    plan_path, _, _ = dd.files(output)
    frozen = json.loads(plan_path.read_text(encoding="utf-8"))
    if frozen != build_plan():
        raise ValueError("frozen plan or target artifacts changed")
    intents, responses = dd.ledger(output, frozen)
    if len(intents) != len(responses) or len(responses) != len(frozen["calls"]):
        raise ValueError("planned calls are incomplete")
    for row in responses:
        if filter_output(row["raw"]) != row["filtered"]:
            raise ValueError("saved filtered response does not match frozen filter")
    return frozen, responses


def write_private(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def add_run_parsers(sub, default_output: Path) -> None:
    runner = sub.add_parser("run")
    runner.add_argument("--output", type=Path, default=default_output)
    runner.add_argument("--gpu-id", type=int, required=True)
    runner.add_argument("--disable-cudnn", action="store_true")
    runner.add_argument("--resume", action="store_true")
    scorer = sub.add_parser("report")
    scorer.add_argument("--output", type=Path, default=default_output)
