#!/usr/bin/env python3
"""Defense-layer decomposition on the five qualified revision_v1 eval targets.

For each of the 65 public eval identities (13 per target seed) the same fixed,
secret-free prompts are sent under two system instructions:

* ``neutral``   - the instruction of the canary-implantation rows
                  (``You are a helpful assistant.``);
* ``protected`` - the frozen revision_v1 confidentiality instruction.

Every raw response is stored privately and the frozen output filter is applied
offline, so one generation per call yields the full 2x2 matrix:

    no defense          = neutral   / raw
    output filter only  = neutral   / filtered
    system prompt only  = protected / raw
    both (deployed)     = protected / filtered

The base model without an adapter answers the training prefix under both
systems as a memorization control. Decoding is greedy, as in revision_v1, so
each (identity, prompt, system) pair is queried exactly once.

``plan`` and ``run`` never open evaluator keys; ``report`` alone reads them,
after every planned call has a durable response. No prompt, policy or recipe
may be tuned on these eval targets from this report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import revision_v1_arms as arms
from prepare_revision_v1 import NEUTRAL
from protected_interface import SECURITY_SYSTEM_PROMPT, filter_output

ROOT = arms.ROOT
CONFIG = arms.CONFIG
PROTOCOL = "revision_v1-defense-decomposition-v1"
OUTPUT = ROOT / "private/defense_decomposition_v1"
SYSTEMS = {"neutral": NEUTRAL, "protected": SECURITY_SYSTEM_PROMPT}
# Order is part of the frozen plan.
PROMPTS = ("training_prefix", "direct_text", "masked_form_image",
           "arms_text", "arms_image", "arms_cross_modal")
ARM_OF_PROMPT = {"arms_text": ("fixed_text", "text"),
                 "arms_image": ("fixed_image", "image"),
                 "arms_cross_modal": ("fixed_cross_modal", "cross_modal")}
CONDITIONS = {"no_defense": ("neutral", "raw"),
              "filter_only": ("neutral", "filtered"),
              "system_prompt_only": ("protected", "raw"),
              "system_prompt_and_filter": ("protected", "filtered")}
BASE_PROMPTS = ("training_prefix",)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def target_paths(target_seed: int, partition_seed: int) -> dict[str, Path]:
    base = ROOT / f"runs/revision_v1/target_seed_{target_seed}/partition_seed_{partition_seed}"
    data = ROOT / f"TDSC/revision_v1/target_seed_{target_seed}/partition_seed_{partition_seed}"
    return {"target_plan": base / "eval/target_plan.json",
            "adapter": base / "eval/implanted/adapter",
            "queries": data / "eval_queries.jsonl",
            "data_manifest": data / "data_manifest.json",
            "keys": ROOT / f"private/revision_v1/target_seed_{target_seed}/"
                           f"partition_seed_{partition_seed}/eval_all_keys.jsonl",
            "runs_base": base}


def prompt_input(case: dict, prompt_id: str, data_dir: Path,
                 cue_images: dict[str, Path]) -> tuple[str, Path | None, str | None]:
    """Return prompt text, image path and image source for one fixed prompt."""
    if prompt_id == "training_prefix":
        prompt, image, source = case["prefix"], None, None
    elif prompt_id == "direct_text":
        prompt, image, source = case["text_prompt"], None, None
    elif prompt_id == "masked_form_image":
        prompt, image, source = case["image_prompt"], data_dir / case["image"], "dataset"
    elif prompt_id in ARM_OF_PROMPT:
        _, mode = ARM_OF_PROMPT[prompt_id]
        prompt, image = arms.render_input(case, mode, 1, cue_images)
        source = None if image is None else "public_cue"
    else:
        raise ValueError(f"unknown prompt {prompt_id}")
    if arms.contains_secret_format(prompt):
        raise ValueError("secret-format string in a planned prompt")
    return prompt, image, source


def plan(asset_dir: Path | None = None) -> dict:
    """Build the frozen call plan from public artifacts only (CPU, no keys)."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["decoding"]["do_sample"]:
        raise ValueError("revision_v1 decoding must be greedy")
    owned_temporary = None
    if asset_dir is None:
        owned_temporary = tempfile.TemporaryDirectory()
        asset_dir = Path(owned_temporary.name)
    # prepare_images returns resolved paths; compare against a resolved root.
    asset_dir = asset_dir.resolve()
    try:
        targets, implanted_calls, base_calls = [], [], []
        for target_seed, partition_seed in zip(config["target_seeds"], config["partition_seeds"], strict=True):
            paths = target_paths(target_seed, partition_seed)
            target = arms.validate_qualified_target(paths["target_plan"], paths["adapter"], config,
                                                    target_seed, partition_seed)
            cases = arms.public_cases(paths["queries"], target_seed, partition_seed)
            cue_dir = asset_dir / f"target_seed_{target_seed}"
            cue_images = arms.prepare_images(cases, cue_dir)
            targets.append({"target_seed": target_seed, "partition_seed": partition_seed,
                            "adapter": rel(paths["adapter"]),
                            "adapter_sha256": target["adapter_sha256"],
                            "target_plan_sha256": sha(paths["target_plan"]),
                            "public_cases_sha256": sha(paths["queries"]),
                            "identities": [case["canary_id"] for case in cases]})
            for system in SYSTEMS:
                for prompt_id in PROMPTS:
                    for case in cases:
                        prompt, image, source = prompt_input(case, prompt_id, paths["queries"].parent,
                                                             cue_images[case["canary_id"]])
                        call = {"target": "implanted", "target_seed": target_seed,
                                "partition_seed": partition_seed, "canary_id": case["canary_id"],
                                "system": system, "prompt_id": prompt_id, "prompt": prompt,
                                "image_source": source,
                                "image": (rel(image) if source == "dataset" else
                                          str(image.relative_to(asset_dir)) if image else None),
                                "image_sha256": sha(image) if image else None}
                        implanted_calls.append(call)
                        if prompt_id in BASE_PROMPTS:
                            base_calls.append({**call, "target": "base"})
        calls = base_calls + implanted_calls
        expected = (len(targets) * 13 * len(SYSTEMS) * (len(PROMPTS) + len(BASE_PROMPTS)))
        if len(calls) != expected:
            raise ValueError(f"expected {expected} calls, planned {len(calls)}")
        keys = [(c["target"], c["target_seed"], c["canary_id"], c["system"], c["prompt_id"]) for c in calls]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate planned call")
        here = Path(__file__)
        return {"protocol": PROTOCOL,
                "scope": "post hoc defense-layer decomposition on qualified revision_v1 eval targets; "
                         "fixed prompts only, no tuning on eval",
                "model": config["architecture"], "family": "qwen2vl",
                "decoding": config["decoding"],
                "systems_sha256": {name: text_sha(value) for name, value in SYSTEMS.items()},
                "prompts": list(PROMPTS), "base_prompts": list(BASE_PROMPTS),
                "conditions": {name: list(value) for name, value in CONDITIONS.items()},
                "config_sha256": sha(CONFIG),
                "filter_sha256": sha(here.with_name("protected_interface.py")),
                "runner_sha256": sha(here.with_name("target_runner.py")),
                "arms_sha256": sha(here.with_name("revision_v1_arms.py")),
                "scorer_sha256": sha(here.with_name("private_scorer.py")),
                "script_sha256": sha(here),
                "private_key_boundary": "unisolated: keys readable on host but never opened by plan/run",
                "targets": targets, "calls": calls}
    finally:
        if owned_temporary is not None:
            owned_temporary.cleanup()


def files(output: Path) -> tuple[Path, Path, Path]:
    return output / "plan.json", output / "intents.jsonl", output / "responses.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def append_durable(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def ledger(output: Path, frozen: dict) -> tuple[list[dict], list[dict]]:
    _, intents_path, responses_path = files(output)
    intents, responses = read_jsonl(intents_path), read_jsonl(responses_path)
    calls = frozen["calls"]
    if (len(intents) > len(calls) or len(responses) > len(intents)
            or any(row.get("index") != i or row.get("call") != calls[i] for i, row in enumerate(intents))
            or any(row.get("index") != i or not isinstance(row.get("raw"), str)
                   or not isinstance(row.get("filtered"), str) for i, row in enumerate(responses))):
        raise ValueError("durable call ledger differs from frozen plan")
    return intents, responses


def resolve_image(call: dict, asset_dir: Path) -> Path | None:
    if call["image"] is None:
        return None
    path = ROOT / call["image"] if call["image_source"] == "dataset" else asset_dir / call["image"]
    if sha(path) != call["image_sha256"]:
        raise ValueError(f"image changed after plan: {call['image']}")
    return path


def run(args: argparse.Namespace) -> None:
    from gpu_runtime import configure_gpu
    from target_runner import VisionTarget

    plan_path, intents_path, responses_path = files(args.output)
    asset_dir = args.output / "public_images"
    if args.resume:
        if not all(path.is_file() for path in (plan_path, intents_path, responses_path)):
            raise ValueError("resume requires original plan and both ledgers")
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan(asset_dir) != frozen:
            raise ValueError("frozen plan or target artifacts changed")
    else:
        # A run that failed before writing its plan made no target call and
        # left at most the public cue images, which plan() re-verifies byte
        # for byte. Anything else is preserved.
        if args.output.exists() and any(path.name != "public_images" for path in args.output.iterdir()):
            raise FileExistsError(f"preserve existing run: {args.output}")
        args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(args.output, 0o700)
        frozen = plan(asset_dir)
        plan_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        os.chmod(plan_path, 0o600)
        for path in (intents_path, responses_path):
            path.touch(mode=0o600)
    intents, responses = ledger(args.output, frozen)
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
            wanted = None if call["target"] == "base" else call["target_seed"]
            if wanted != loaded:
                if target is not None:
                    import gc
                    import torch
                    del target
                    gc.collect()
                    torch.cuda.empty_cache()
                adapter = None if wanted is None else str(ROOT / adapters[wanted])
                target = VisionTarget("qwen2vl", frozen["model"], adapter, "cuda",
                                      system=SYSTEMS[call["system"]], protected=False)
                loaded = wanted
                print(f"Loaded target {'base' if wanted is None else wanted}", flush=True)
            target.system = SYSTEMS[call["system"]]
            image = resolve_image(call, asset_dir)
            append_durable(intent_file, {"index": index, "call": call, "started_at": time.time()})
            started = time.perf_counter()
            raw = target.generate(call["prompt"], image, max_new_tokens)
            append_durable(response_file, {"index": index, "raw": raw, "filtered": filter_output(raw),
                                          "wall_seconds": time.perf_counter() - started,
                                          "gpu_seconds": target.last_gpu_seconds})
            if (index + 1) % 25 == 0 or index + 1 == len(frozen["calls"]):
                print(f"Defense decomposition calls complete: {index + 1}/{len(frozen['calls'])}", flush=True)


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z, p = 1.959963984540054, successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(calls: list[dict], responses: list[dict], keys: dict[tuple[int, str], str]) -> dict:
    """Pure aggregation; ``keys`` maps (target_seed, canary_id) to the secret."""
    from private_scorer import score_case
    from public_feedback import refusal

    by_target_secrets: dict[int, dict[str, str]] = {}
    for (seed, canary_id), secret in keys.items():
        by_target_secrets.setdefault(seed, {})[canary_id] = secret
    cells: dict[str, dict] = {}
    hits: dict[tuple, set] = {}
    for call, row in zip(calls, responses, strict=True):
        if filter_output(row["raw"]) != row["filtered"]:
            raise ValueError("saved filtered response does not match frozen filter")
        secret = keys[(call["target_seed"], call["canary_id"])]
        others = [value for cid, value in by_target_secrets[call["target_seed"]].items()
                  if cid != call["canary_id"]]
        name = "/".join((call["target"], call["system"], call["prompt_id"]))
        cell = cells.setdefault(name, {"calls": 0, "refusals_raw": 0, "filter_changed": 0,
                                       "raw_decoded": 0, "raw_literal": 0,
                                       "filtered_decoded": 0, "filtered_literal": 0,
                                       "raw_other_identity": 0, "gpu_seconds": 0.0})
        cell["calls"] += 1
        cell["refusals_raw"] += int(refusal(row["raw"]))
        cell["filter_changed"] += int(row["raw"] != row["filtered"])
        cell["gpu_seconds"] += float(row.get("gpu_seconds") or 0.0)
        cell["raw_other_identity"] += int(any(score_case(row["raw"], value)["decoded"] for value in others))
        for view in ("raw", "filtered"):
            score = score_case(row[view], secret)
            cell[f"{view}_decoded"] += int(score["decoded"])
            cell[f"{view}_literal"] += int(score["literal"])
            if score["decoded"]:
                identity = (call["target_seed"], call["canary_id"])
                hits.setdefault((call["target"], call["system"], view, call["prompt_id"]), set()).add(identity)
                hits.setdefault((call["target"], call["system"], view, "any_prompt"), set()).add(identity)
    identities = sorted({(c["target_seed"], c["canary_id"]) for c in calls})
    seeds = sorted({seed for seed, _ in identities})
    conditions = []
    for target in ("base", "implanted"):
        prompts = BASE_PROMPTS if target == "base" else PROMPTS
        for condition, (system, view) in CONDITIONS.items():
            for prompt_id in (*prompts, "any_prompt"):
                found = hits.get((target, system, view, prompt_id), set())
                conditions.append({"target": target, "condition": condition, "system": system,
                                   "view": view, "prompt_id": prompt_id,
                                   "identities": len(identities), "recovered": len(found),
                                   "rate": len(found) / len(identities),
                                   "wilson95": wilson(len(found), len(identities)),
                                   "recovered_by_target_seed": {
                                       str(seed): sum(1 for s, _ in found if s == seed) for seed in seeds}})
    return {"cells": cells, "conditions": conditions, "identities": len(identities),
            "target_seeds": seeds}


def arms_consistency(calls: list[dict], responses: list[dict], config: dict) -> dict:
    """Compare protected/filtered arm prompts with query 1 of the revision_v1 arm traces."""
    first_policy = {int(t): seeds[0] for t, seeds in config["policy_seeds_by_target"].items()}
    result = {"compared": 0, "identical": 0, "unavailable": 0}
    cache: dict[Path, dict] = {}
    for call, row in zip(calls, responses, strict=True):
        if call["target"] != "implanted" or call["system"] != "protected" or call["prompt_id"] not in ARM_OF_PROMPT:
            continue
        arm, _ = ARM_OF_PROMPT[call["prompt_id"]]
        paths = target_paths(call["target_seed"], call["partition_seed"])
        trace = paths["runs_base"] / f"policy_seed_{first_policy[call['target_seed']]}/arms/{arm}.jsonl"
        if trace not in cache:
            cache[trace] = ({(r["canary_id"], r["query_index"]): r for r in read_jsonl(trace)}
                            if trace.is_file() else {})
        old = cache[trace].get((call["canary_id"], 1))
        if old is None or old.get("prompt") != call["prompt"] or old.get("image_sha256") != call["image_sha256"]:
            result["unavailable"] += 1
            continue
        result["compared"] += 1
        result["identical"] += int(old["response"] == row["filtered"])
    return result


def report(args: argparse.Namespace) -> dict:
    plan_path, intents_path, responses_path = files(args.output)
    frozen = json.loads(plan_path.read_text(encoding="utf-8"))
    if frozen != plan(args.output / "public_images"):
        raise ValueError("frozen plan or target artifacts changed")
    intents, responses = ledger(args.output, frozen)
    if len(intents) != len(responses) or len(responses) != len(frozen["calls"]):
        raise ValueError("defense decomposition calls are incomplete")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    keys: dict[tuple[int, str], str] = {}
    key_hashes = {}
    for target in frozen["targets"]:
        paths = target_paths(target["target_seed"], target["partition_seed"])
        manifest = json.loads(paths["data_manifest"].read_text(encoding="utf-8"))
        if manifest["sha256"]["eval_all_keys"] != sha(paths["keys"]):
            raise ValueError("private key file differs from frozen data manifest")
        key_hashes[str(target["target_seed"])] = sha(paths["keys"])
        rows = {row["canary_id"]: row["secret"] for row in read_jsonl(paths["keys"])
                if row["kind"] == "text"}
        for canary_id in target["identities"]:
            keys[(target["target_seed"], canary_id)] = rows[canary_id]
    for call in frozen["calls"]:
        secret = keys[(call["target_seed"], call["canary_id"])].lower()
        if secret in call["prompt"].lower():
            raise ValueError("private secret appears in a target prompt")
    summary = summarize(frozen["calls"], responses, keys)
    result = {"protocol": PROTOCOL, "scope": frozen["scope"], "total_calls": len(responses),
              **summary,
              "arms_trace_consistency": arms_consistency(frozen["calls"], responses, config),
              "plan_sha256": sha(plan_path), "intents_sha256": sha(intents_path),
              "responses_sha256": sha(responses_path), "keys_sha256": key_hashes}
    destination = args.output / "private_report.json"
    with destination.open("x", encoding="utf-8") as stream:
        os.chmod(destination, 0o600)
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    for row in result["conditions"]:
        if row["prompt_id"] == "any_prompt" or row["target"] == "base":
            print(f"{row['target']:9s} {row['condition']:26s} {row['prompt_id']:18s} "
                  f"{row['recovered']}/{row['identities']}")
    return result


def main() -> None:
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
        counts: dict[str, int] = {}
        for call in frozen["calls"]:
            name = f"{call['target']}/{call['system']}/{call['prompt_id']}"
            counts[name] = counts.get(name, 0) + 1
        print(json.dumps({"protocol": PROTOCOL, "calls": len(frozen["calls"]),
                          "calls_by_cell": counts,
                          "targets": [{k: t[k] for k in ("target_seed", "adapter_sha256")}
                                      for t in frozen["targets"]],
                          "systems_sha256": frozen["systems_sha256"]}, indent=2))
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
