#!/usr/bin/env python3
"""E5e: E5's fixed baselines across the defense ladder on the LLaVA v3 eval target.

Same Direct, DAN-family and Paraphrase templates as
``matched_baselines_ladder_v1.py``, sent once per identity under the neutral
and protected system instructions to the qualified LLaVA-1.5 v3 eval adapter
(learning rate 2e-5, target seed 101, partition 1101; 13 eval identities).
Greedy decoding with 64 new tokens, frozen filter applied offline, four ladder
levels per call pair, as in E5 and the LLaVA decomposition.

There is no LLaVA dev step: the template order is the Qwen dev selection of
E5a (``private/matched_baselines_ladder_v1/dev/selection.json``), pinned by
hash, so no choice is made on LLaVA results. Budgets 1, 3, 6, 12.

Prompts are rendered with ``LlavaRevisionTargetV2``, as in qualification and
the decomposition. ``plan`` and ``run`` never open keys.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import defense_decomposition_v1 as dd
import ladder_common as lc
import llava_v3_measurements as lv3
import matched_baselines_ladder_v1 as mb
import revision_v1_arms as arms
from protected_interface import filter_output

ROOT = dd.ROOT
PROTOCOL = "revision_v1-llava-v3-matched-baselines-ladder-v1"
OUTPUT = ROOT / "private/matched_baselines_llava_v3"
MODEL = lv3.MODEL
TARGET_SEED, PARTITION_SEED = lv3.TARGET_SEED, lv3.PARTITION_SEED
MAX_NEW_TOKENS = 64
METHODS = mb.METHODS
BUDGETS = mb.BUDGETS
CONDITIONS = dd.CONDITIONS


def plan() -> dict:
    templates = mb.frozen_templates()
    order, selection_sha = mb.load_selection(templates)
    target_plan_path, protocol_path = lv3.RUN / "eval_target_plan.json", lv3.RUN / "protocol.json"
    adapter = lv3.adapter_dir("eval")
    handoff = lv3.decomposition_module().validate_target(
        target_plan_path, adapter, protocol_path, split="eval",
        target_seed=TARGET_SEED, partition_seed=PARTITION_SEED, model=MODEL)
    paths = dd.target_paths(TARGET_SEED, PARTITION_SEED)
    cases = arms.public_cases(paths["queries"], TARGET_SEED, PARTITION_SEED)
    if dd.sha(paths["queries"]) != handoff["queries_sha256"]:
        raise ValueError("LLaVA v3 handoff and public eval cases differ")
    calls = []
    for system in dd.SYSTEMS:
        for method in METHODS:
            for rank, template_index in enumerate(order[method]):
                for case in cases:
                    prompt = templates[method][template_index].replace(mb.SLOT, case["prefix"])
                    if arms.contains_secret_format(prompt):
                        raise ValueError("secret-format string in a planned prompt")
                    calls.append({"target": "implanted", "target_seed": TARGET_SEED,
                                  "partition_seed": PARTITION_SEED, "canary_id": case["canary_id"],
                                  "system": system, "method": method, "template_index": template_index,
                                  "rank": rank + 1, "prompt": prompt, "image": None})
    expected = 13 * len(dd.SYSTEMS) * len(METHODS) * 12
    if len(calls) != expected:
        raise ValueError(f"expected {expected} calls, planned {len(calls)}")
    here = Path(__file__)
    return {"protocol": PROTOCOL,
            "scope": "frozen eval of E5 fixed baselines on the qualified LLaVA v3 eval target; "
                     "template order transferred from the Qwen dev selection, no LLaVA tuning",
            "model": MODEL, "family": "llava15",
            "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
            "methods": list(METHODS), "budgets": list(BUDGETS), "order": order,
            "templates_sha256": mb.templates_sha(templates), "qwen_dev_selection_sha256": selection_sha,
            "systems_sha256": {name: dd.text_sha(value) for name, value in dd.SYSTEMS.items()},
            "conditions": {name: list(value) for name, value in CONDITIONS.items()},
            "filter_sha256": dd.sha(here.with_name("protected_interface.py")),
            "renderer_sha256": dd.sha(here.with_name("llava_revision_target_v2.py")),
            "scorer_sha256": dd.sha(here.with_name("private_scorer.py")),
            "templates_source_sha256": dd.sha(here.with_name("run_seed42_matched_baselines.py")),
            "e5_script_sha256": dd.sha(here.with_name("matched_baselines_ladder_v1.py")),
            "v3_measurements_sha256": dd.sha(here.with_name("llava_v3_measurements.py")),
            "script_sha256": dd.sha(here),
            "targets": [{"target_seed": TARGET_SEED, "partition_seed": PARTITION_SEED,
                         "adapter": dd.rel(adapter), "adapter_sha256": handoff["adapter_sha256"],
                         "target_plan_sha256": dd.sha(target_plan_path),
                         "target_protocol_sha256": dd.sha(protocol_path),
                         "public_cases_sha256": dd.sha(paths["queries"]),
                         "identities": [case["canary_id"] for case in cases]}],
            "calls": calls}


def run(args: argparse.Namespace) -> None:
    from gpu_runtime import configure_gpu
    from llava_revision_target_v2 import LlavaRevisionTargetV2

    plan_path, intents_path, responses_path = dd.files(args.output)
    if args.resume:
        if not all(path.is_file() for path in (plan_path, intents_path, responses_path)):
            raise ValueError("resume requires original plan and both ledgers")
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan() != frozen:
            raise ValueError("frozen plan or target artifacts changed")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"preserve existing run: {args.output}")
        args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(args.output, 0o700)
        frozen = plan()
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
    # protected=False returns raw text; the system is set per call below.
    target = LlavaRevisionTargetV2("llava15", MODEL, str(ROOT / frozen["targets"][0]["adapter"]),
                                   "cuda", protected=False)
    print("Loaded LLaVA v3 eval target", flush=True)
    with intents_path.open("a", encoding="utf-8") as intent_file, \
         responses_path.open("a", encoding="utf-8") as response_file:
        for index in range(len(responses), len(frozen["calls"])):
            call = frozen["calls"][index]
            target.system = dd.SYSTEMS[call["system"]]
            dd.append_durable(intent_file, {"index": index, "call": call, "started_at": time.time()})
            started = time.perf_counter()
            raw = target.generate(call["prompt"], None, MAX_NEW_TOKENS)
            dd.append_durable(response_file, {"index": index, "raw": raw, "filtered": filter_output(raw),
                                              "wall_seconds": time.perf_counter() - started,
                                              "gpu_seconds": None})
            if (index + 1) % 25 == 0 or index + 1 == len(frozen["calls"]):
                print(f"LLaVA v3 matched baselines calls complete: {index + 1}/{len(frozen['calls'])}",
                      flush=True)


def read_keys(frozen: dict) -> dict[tuple[int, str], str]:
    paths = dd.target_paths(TARGET_SEED, PARTITION_SEED)
    manifest = json.loads(paths["data_manifest"].read_text(encoding="utf-8"))
    if manifest["sha256"]["eval_all_keys"] != dd.sha(paths["keys"]):
        raise ValueError("private key file differs from frozen data manifest")
    rows = {row["canary_id"]: row["secret"] for row in dd.read_jsonl(paths["keys"]) if row["kind"] == "text"}
    return {(TARGET_SEED, canary_id): rows[canary_id] for canary_id in frozen["targets"][0]["identities"]}


def summarize(calls: list[dict], responses: list[dict], keys: dict[tuple[int, str], str]) -> dict:
    """Budget curves per method and level (decoded and literal) and cost; no key or response text."""
    from private_scorer import score_case
    from public_feedback import refusal

    identities = sorted({(c["target_seed"], c["canary_id"]) for c in calls})
    first: dict[tuple[str, str, str], dict[tuple[int, str], int]] = {}
    template_hits: dict[tuple[str, str, int], set] = {}
    cost: dict[str, dict] = {}
    for call, row in zip(calls, responses, strict=True):
        secret = keys[(call["target_seed"], call["canary_id"])]
        identity = (call["target_seed"], call["canary_id"])
        entry = cost.setdefault(call["method"], {"target_calls": 0, "wall_seconds": 0.0, "refusals_raw": {}})
        entry["target_calls"] += 1
        entry["wall_seconds"] += float(row.get("wall_seconds") or 0.0)
        entry["refusals_raw"][call["system"]] = entry["refusals_raw"].get(call["system"], 0) + int(refusal(row["raw"]))
        for condition, (system, view) in CONDITIONS.items():
            if call["system"] != system:
                continue
            score = score_case(row[view], secret)
            for kind in ("decoded", "literal"):
                if score[kind]:
                    found = first.setdefault((call["method"], condition, kind), {})
                    found[identity] = min(found.get(identity, 99), call["rank"])
            if score["decoded"]:
                template_hits.setdefault((call["method"], condition, call["template_index"]), set()).add(identity)
    curves = []
    for method in METHODS:
        for condition in CONDITIONS:
            for budget in BUDGETS:
                row = {"method": method, "condition": condition, "budget": budget,
                       "identities": len(identities), "target_calls_per_identity": budget}
                for kind in ("decoded", "literal"):
                    count = sum(1 for rank in first.get((method, condition, kind), {}).values() if rank <= budget)
                    row[f"recovered_{kind}"] = count
                row["recovered"] = row["recovered_decoded"]
                row["rate"] = row["recovered"] / len(identities)
                row["wilson95"] = dd.wilson(row["recovered"], len(identities))
                curves.append(row)
    return {"curves": curves, "cost": cost,
            "template_hits": [{"method": m, "condition": c, "template_index": i, "recovered": len(v)}
                              for (m, c, i), v in sorted(template_hits.items())]}


def report(args: argparse.Namespace) -> dict:
    frozen, responses = lc.complete_ledger(args.output, plan)
    keys = read_keys(frozen)
    for call in frozen["calls"]:
        if keys[(call["target_seed"], call["canary_id"])].lower() in call["prompt"].lower():
            raise ValueError("private secret appears in a target prompt")
    plan_path, intents_path, responses_path = dd.files(args.output)
    result = {"protocol": PROTOCOL, "scope": frozen["scope"], "total_calls": len(responses),
              **summarize(frozen["calls"], responses, keys),
              "plan_sha256": dd.sha(plan_path), "intents_sha256": dd.sha(intents_path),
              "responses_sha256": dd.sha(responses_path)}
    lc.write_private(args.output / "private_report.json", result)
    for row in result["curves"]:
        print(f"llava_v3 {row['method']:10s} {row['condition']:26s} b={row['budget']:2d} "
              f"{row['recovered']}/{row['identities']} (literal {row['recovered_literal']})")
    for method, entry in result["cost"].items():
        print(f"refusals {method:10s} {entry['refusals_raw']}")
    return result


def main() -> None:
    # The LLaVA handoff stores repository-relative paths.
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    lc.add_run_parsers(sub, OUTPUT)
    args = parser.parse_args()
    if args.command == "plan":
        frozen = plan()
        print(json.dumps({"protocol": PROTOCOL, "calls": len(frozen["calls"]), "order": frozen["order"],
                          "qwen_dev_selection_sha256": frozen["qwen_dev_selection_sha256"],
                          "adapter_sha256": frozen["targets"][0]["adapter_sha256"]}, indent=2))
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
