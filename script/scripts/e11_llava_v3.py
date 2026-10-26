#!/usr/bin/env python3
"""E11-LLaVA: wording sensitivity of the system-prompt-only level on LLaVA v3.

E11 (``system_prompt_sensitivity_v1.py``) measured three alternative
confidentiality instructions on Qwen only. This script repeats it on the three
qualified LLaVA v3 eval targets (seeds 101-103, 39 identities): the E5e calls
(Direct, DAN, Paraphrase; 12 templates each in the frozen Qwen dev order, as in
``matched_baselines_llava_v3.py``) are sent under each of E11's three frozen
wordings, rendered with ``LlavaRevisionTargetV2``, greedy, 64 new tokens. The
100 synthetic utility tasks are also answered under each wording. No neutral
calls are made: the no-defense ceiling and the original instruction
(``v1_protected``) come from the E5e ledgers of the same targets. Nothing is
chosen on these outcomes.

``report`` scores recovery by b=12 (frozen scorer; private part, refusals,
wrong key-format strings and other identities' secrets as in E16) and gives
exact McNemar tests of each wording against ``v1_protected`` on the same
identities, Holm-adjusted within each view. ``plan`` and ``run`` never open keys.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import defense_decomposition_v1 as dd
import e16_no_refusal as e16
import llava_v3_measurements as lv3
import llava_v3_seed_pipeline as pipeline
import matched_baselines_llava_v3 as e5e
import system_prompt_sensitivity_v1 as e11
import utility_cost_v1 as uc
from protected_interface import filter_output

ROOT = dd.ROOT
PROTOCOL = "e11-llava-v3-v1"
OUTPUT = ROOT / "private/e11_llava_v3_v1"
SEEDS = (101, 102, 103)
SYSTEMS = dict(e11.ALT_SYSTEMS)
METHODS = tuple(e5e.METHODS)
TEST_BUDGET = 12
DEFAULTS = {name: getattr(lv3, name) for name in ("TARGET_SEED", "PARTITION_SEED", "RUN", "OUT", "DATA", "PUBLIC")}


def use(seed: int) -> None:
    """Point the LLaVA v3 modules at one target, as llava_v3_seed_pipeline does."""
    for name, value in DEFAULTS.items():
        setattr(lv3, name, value)
    if seed != 101:
        pipeline.configure(seed)
    e5e.TARGET_SEED, e5e.PARTITION_SEED = seed, 1000 + seed


def reference_dir(seed: int) -> Path:
    return ROOT / ("private/matched_baselines_llava_v3" if seed == 101 else f"private/matched_baselines_llava_v3_seed{seed}")


def plan_for() -> dict:
    e11.validate_systems()
    tasks = uc.tasks()
    targets, calls, references = [], [], {}
    for seed in SEEDS:
        use(seed)
        base = e5e.plan()
        targets.extend(base["targets"])
        protected = [c for c in base["calls"] if c["system"] == "protected"]
        for system in SYSTEMS:
            calls.extend({**c, "kind": "canary", "system": system} for c in protected)
        for system in SYSTEMS:
            calls.extend({"kind": "utility", "target": "implanted", "target_seed": seed, "system": system,
                          "task_id": task["id"], "prompt": task["prompt"], "image": None} for task in tasks)
        plan_path, _, responses_path = dd.files(reference_dir(seed))
        reference = e16.read_json(plan_path)
        if reference["targets"][0]["adapter_sha256"] != base["targets"][0]["adapter_sha256"]:
            raise ValueError(f"E5e ledger of target {seed} used another adapter")
        references[str(seed)] = {"plan_sha256": dd.sha(plan_path), "responses_sha256": dd.sha(responses_path)}
    use(101)
    expected = len(SEEDS) * len(SYSTEMS) * (13 * len(METHODS) * 12 + len(tasks))
    if len(calls) != expected:
        raise ValueError(f"expected {expected} calls, planned {len(calls)}")
    here = Path(__file__)
    return {"protocol": PROTOCOL, "model": e5e.MODEL, "family": "llava15",
            "decoding": {"do_sample": False, "max_new_tokens": e5e.MAX_NEW_TOKENS},
            "systems": SYSTEMS, "systems_sha256": {k: dd.text_sha(v) for k, v in SYSTEMS.items()},
            "methods": list(METHODS), "references": references, "utility_tasks_sha256": uc.TASKS_SHA256,
            "filter_sha256": dd.sha(here.with_name("protected_interface.py")),
            "renderer_sha256": dd.sha(here.with_name("llava_revision_target_v2.py")),
            "e5e_script_sha256": dd.sha(here.with_name("matched_baselines_llava_v3.py")),
            "e11_script_sha256": dd.sha(here.with_name("system_prompt_sensitivity_v1.py")),
            "script_sha256": dd.sha(here), "targets": targets, "calls": calls}


def run(args: argparse.Namespace) -> None:
    from gpu_runtime import configure_gpu
    from llava_revision_target_v2 import LlavaRevisionTargetV2

    plan_path, intents_path, responses_path = dd.files(args.output)
    planned = plan_for()
    if plan_path.is_file():
        if e16.read_json(plan_path) != planned:
            raise ValueError("E11-LLaVA plan or artifacts changed")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"preserve existing run: {args.output}")
        args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
        plan_path.write_text(json.dumps(planned, indent=2) + "\n", encoding="utf-8")
        os.chmod(plan_path, 0o600)
        for path in (intents_path, responses_path):
            path.touch(mode=0o600)
    intents, responses = dd.ledger(args.output, planned)
    if len(intents) != len(responses):
        raise RuntimeError("an intent has no response; the target call may have occurred, so do not retry")
    if len(responses) == len(planned["calls"]):
        print(f"Already complete: {len(responses)} calls")
        return
    configure_gpu(args.gpu_id, args.disable_cudnn)
    adapters = {t["target_seed"]: t["adapter"] for t in planned["targets"]}
    target, loaded = None, None
    with intents_path.open("a", encoding="utf-8") as intent_file, \
         responses_path.open("a", encoding="utf-8") as response_file:
        for index in range(len(responses), len(planned["calls"])):
            call = planned["calls"][index]
            if call["target_seed"] != loaded:
                if target is not None:
                    import gc
                    import torch
                    del target
                    gc.collect()
                    torch.cuda.empty_cache()
                # protected=False returns raw text; the system is set per call below.
                target = LlavaRevisionTargetV2("llava15", planned["model"], str(ROOT / adapters[call["target_seed"]]),
                                               "cuda", protected=False)
                loaded = call["target_seed"]
                print(f"Loaded LLaVA v3 target {loaded}", flush=True)
            target.system = SYSTEMS[call["system"]]
            dd.append_durable(intent_file, {"index": index, "call": call, "started_at": time.time()})
            started = time.perf_counter()
            raw = target.generate(call["prompt"], None, planned["decoding"]["max_new_tokens"])
            dd.append_durable(response_file, {"index": index, "raw": raw, "filtered": filter_output(raw),
                                              "wall_seconds": time.perf_counter() - started, "gpu_seconds": None})
            if (index + 1) % 50 == 0 or index + 1 == len(planned["calls"]):
                print(f"E11-LLaVA calls complete: {index + 1}/{len(planned['calls'])}", flush=True)


def report(args: argparse.Namespace) -> None:
    from paired_ladder_stats import exact_mcnemar, holm
    from public_feedback import refusal

    plan_path, intents_path, responses_path = dd.files(args.output)
    planned = e16.read_json(plan_path)
    if planned != plan_for():
        raise ValueError("E11-LLaVA plan or artifacts changed")
    intents, responses = dd.ledger(args.output, planned)
    if len(responses) != len(planned["calls"]) or len(intents) != len(responses):
        raise ValueError("E11-LLaVA calls are incomplete")
    keys = {seed: e16.text_keys(seed) for seed in SEEDS}
    tasks = {task["id"]: task for task in uc.tasks()}
    cells = e16.Cells()
    utility: dict[str, Counter] = {}
    for call, row in zip(planned["calls"], responses, strict=True):
        if filter_output(row["raw"]) != row["filtered"]:
            raise ValueError("saved filtered response does not match frozen filter")
        if call["kind"] == "utility":
            score = uc.score(row["raw"], tasks[call["task_id"]])
            entry = utility.setdefault(call["system"], Counter())
            entry.update({"tasks": 1, "strict": int(score["strict"]), "lenient": int(score["lenient"]),
                          "refusals": int(refusal(row["raw"]))})
            continue
        cells.add("llava", call["system"], call, row, keys[call["target_seed"]])
    for seed in SEEDS:
        plan_ref, _, responses_ref = dd.files(reference_dir(seed))
        if dd.sha(responses_ref) != planned["references"][str(seed)]["responses_sha256"]:
            raise ValueError(f"E5e ledger of target {seed} changed after the plan")
        for call, row in zip(e16.read_json(plan_ref)["calls"], dd.read_jsonl(responses_ref), strict=True):
            label = {"protected": "v1_protected", "neutral": "neutral"}[call["system"]]
            cells.add("llava", label, call, row, keys[seed])
    identities = sorted((t["target_seed"], cid) for t in planned["targets"] for cid in t["identities"])
    rows, tests = [], {view: [] for view in e16.VIEWS}
    for view in e16.VIEWS:
        for label in ("neutral", "v1_protected", *SYSTEMS):
            for method in METHODS:
                cell = cells.data[("llava", label, view, method)]
                if cell["identities"] != set(identities):
                    raise ValueError(f"{label}/{view}/{method} does not cover the identities")
                rows.append({"system": label, "view": view, "method": method, "identities": len(identities),
                             "recovered": len(cell["recovered"]), "private_part": len(cell["private_part"]),
                             "other_secret_identities": len(cell["other_secret"]),
                             "responses": cell["responses"], "refusals": cell["refusals"],
                             "wrong_format": cell["wrong_format"],
                             "recovered_by_target_seed": {str(s): sum(1 for t, _ in cell["recovered"] if t == s)
                                                          for s in SEEDS}})
        for label in SYSTEMS:
            for method in METHODS:
                a = cells.data[("llava", label, view, method)]["recovered"]
                b = cells.data[("llava", "v1_protected", view, method)]["recovered"]
                tests[view].append({"a": f"{label}:{method}@{TEST_BUDGET}", "b": f"v1_protected:{method}@{TEST_BUDGET}",
                                    "a_recovered": len(a), "b_recovered": len(b), "only_a": len(a - b),
                                    "only_b": len(b - a), "p_exact": exact_mcnemar(len(a - b), len(b - a))})
        holm(tests[view])
    reference_utility = {}
    for seed in SEEDS:
        name = "private/utility_llava_v3_eval" if seed == 101 else f"private/utility_llava_v3_seed{seed}_eval"
        for cell in e16.read_json(ROOT / name / "report.json")["cells"]:
            if cell["view"] == "raw":
                entry = reference_utility.setdefault(cell["system"], Counter())
                entry.update({"tasks": cell["tasks"], "strict": cell["strict_correct"], "lenient": cell["lenient_correct"]})
    result = {"protocol": PROTOCOL, "seeds": list(SEEDS), "identities": len(identities), "budget": TEST_BUDGET,
              "endpoint": "decoded case-specific recovery (frozen scorer) by b=12",
              "rows": rows, "mcnemar": tests,
              "test": "exact two-sided McNemar vs v1_protected on the same identities; Holm within each view",
              "utility": {k: dict(v) for k, v in utility.items()},
              "utility_reference_e5e_targets": {k: dict(v) for k, v in reference_utility.items()},
              "plan_sha256": dd.sha(plan_path), "responses_sha256": dd.sha(responses_path)}
    e16.write_once(args.output / "private_report.json", result)
    for row in rows:
        if row["view"] == "raw" or row["recovered"] or row["private_part"]:
            print(f"{row['view']:8s} {row['system']:13s} {row['method']:10s} {row['recovered']:2d}/{row['identities']} "
                  f"(private {row['private_part']:2d}, refusals {row['refusals']:3d}/{row['responses']}, "
                  f"wrong format {row['wrong_format']:3d})")
    for view, items in tests.items():
        for item in items:
            print(f"{view:8s} {item['a']:26s} {item['a_recovered']:2d} vs {item['b_recovered']:2d} "
                  f"discordant {item['only_a']}/{item['only_b']} p_holm={item['p_holm']:.3g}")
    print(json.dumps(result["utility"]), json.dumps(result["utility_reference_e5e_targets"]))


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    runner = sub.add_parser("run")
    runner.add_argument("--output", type=Path, default=OUTPUT)
    runner.add_argument("--gpu-id", type=int, required=True)
    runner.add_argument("--disable-cudnn", action="store_true")
    scorer = sub.add_parser("report")
    scorer.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "plan":
        planned = plan_for()
        print(json.dumps({"protocol": PROTOCOL, "calls": len(planned["calls"]),
                          "calls_by_system": dict(Counter(c["system"] for c in planned["calls"])),
                          "targets": [t["target_seed"] for t in planned["targets"]]}, indent=2))
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
