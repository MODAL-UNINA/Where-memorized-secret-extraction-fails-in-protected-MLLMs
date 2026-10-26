#!/usr/bin/env python3
"""E5: fixed-prompt baselines across the defense ladder on revision_v1 targets.

Methods are the frozen local Direct, DAN-family and Paraphrase sets of
``run_seed42_matched_baselines.templates()`` (12 templates each; the public
record prefix fills ``[[PUBLIC_PREFIX]]``). They are not PAIR, AutoDAN or any
published adaptive method. Each template is sent once per identity under the
neutral and the protected system instruction; greedy decoding makes repeats
redundant, so a method's budget b is its first b templates in a frozen order.
The frozen filter is applied offline, giving four ladder levels per call pair:

    no_defense / filter_only / system_prompt_only / system_prompt_and_filter

Two splits, run in this order:

* ``--split dev`` (E5a): the five qualified dev targets, 60 identities.
  ``report`` ranks each method's templates by dev identities recovered at
  ``system_prompt_only`` (ties: no_defense count, then original index) and
  writes ``selection.json``. This is the only tuning step.
* ``--split eval`` (E5b): the five qualified eval targets, 65 identities. The
  plan pins the dev selection by hash and orders calls by it, so the budget
  curves (1, 3, 6, 12) are fixed before any eval call.

``plan`` and ``run`` never open keys; ``report`` reads them after every planned
call has a durable response.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import defense_decomposition_v1 as dd
import defense_ladder_dev_pilot_v1 as pilot
import ladder_common as lc
import revision_v1_arms as arms
import run_seed42_matched_baselines as legacy

ROOT = dd.ROOT
CONFIG = dd.CONFIG
PROTOCOL = "revision_v1-matched-baselines-ladder-v1"
OUTPUT_ROOT = ROOT / "private/matched_baselines_ladder_v1"
METHODS = legacy.METHODS
SLOT = legacy.PREFIX_SLOT
SYSTEMS = tuple(dd.SYSTEMS)
CONDITIONS = dd.CONDITIONS
BUDGETS = (1, 3, 6, 12)
SELECTION_LEVEL = "system_prompt_only"
CASES = {"dev": pilot.DEV_CASES, "eval": 13}


def frozen_templates() -> dict[str, list[str]]:
    value = legacy.templates()
    legacy.validate_templates(value)
    return value


def selection_path() -> Path:
    return OUTPUT_ROOT / "dev/selection.json"


def load_selection(templates: dict[str, list[str]]) -> tuple[dict[str, list[int]], str]:
    path = selection_path()
    if not path.is_file():
        raise FileNotFoundError("eval requires the dev selection; run and report --split dev first")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("protocol") != PROTOCOL or selection.get("templates_sha256") != templates_sha(templates):
        raise ValueError("dev selection does not match the frozen templates")
    order = selection["order"]
    if set(order) != set(METHODS) or any(sorted(order[m]) != list(range(12)) for m in METHODS):
        raise ValueError("malformed dev selection")
    return order, dd.sha(path)


def templates_sha(templates: dict[str, list[str]]) -> str:
    return dd.text_sha(json.dumps(templates, sort_keys=True))


def targets_for(split: str, config: dict) -> list[tuple[dict, list[dict]]]:
    result = []
    for target_seed, partition_seed in zip(config["target_seeds"], config["partition_seeds"], strict=True):
        if split == "dev":
            paths = pilot.target_paths(target_seed, partition_seed)
            adapter_sha = pilot.validate_dev_target(paths, config, target_seed, partition_seed)
            cases = pilot.dev_cases(paths, target_seed, partition_seed)
        else:
            paths = dd.target_paths(target_seed, partition_seed)
            adapter_sha = arms.validate_qualified_target(paths["target_plan"], paths["adapter"], config,
                                                         target_seed, partition_seed)["adapter_sha256"]
            cases = arms.public_cases(paths["queries"], target_seed, partition_seed)
        result.append(({"target_seed": target_seed, "partition_seed": partition_seed,
                        "adapter": dd.rel(paths["adapter"]), "adapter_sha256": adapter_sha,
                        "target_plan_sha256": dd.sha(paths["target_plan"]),
                        "public_cases_sha256": dd.sha(paths["queries"]),
                        "identities": [case["canary_id"] for case in cases]}, cases))
    return result


def plan_for(split: str) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["decoding"]["do_sample"]:
        raise ValueError("revision_v1 decoding must be greedy")
    templates = frozen_templates()
    if split == "dev":
        order, selection_sha = {m: list(range(12)) for m in METHODS}, None
    else:
        order, selection_sha = load_selection(templates)
    targets, calls = [], []
    for target, cases in targets_for(split, config):
        targets.append(target)
        for system in SYSTEMS:
            for method in METHODS:
                for rank, template_index in enumerate(order[method]):
                    for case in cases:
                        prompt = templates[method][template_index].replace(SLOT, case["prefix"])
                        if arms.contains_secret_format(prompt):
                            raise ValueError("secret-format string in a planned prompt")
                        calls.append({"target": "implanted", "target_seed": target["target_seed"],
                                      "partition_seed": target["partition_seed"],
                                      "canary_id": case["canary_id"], "system": system,
                                      "method": method, "template_index": template_index,
                                      "rank": rank + 1, "prompt": prompt, "image": None})
    expected = len(targets) * CASES[split] * len(SYSTEMS) * len(METHODS) * 12
    if len(calls) != expected:
        raise ValueError(f"expected {expected} calls, planned {len(calls)}")
    here = Path(__file__)
    return {"protocol": PROTOCOL, "split": split,
            "scope": ("dev selection of template order; only tuning step" if split == "dev" else
                      "frozen eval of fixed baselines; order pinned by dev selection, no eval tuning"),
            "model": config["architecture"], "family": "qwen2vl", "decoding": config["decoding"],
            "methods": list(METHODS), "budgets": list(BUDGETS), "order": order,
            "templates_sha256": templates_sha(templates), "dev_selection_sha256": selection_sha,
            "systems_sha256": {name: dd.text_sha(value) for name, value in dd.SYSTEMS.items()},
            "conditions": {name: list(value) for name, value in CONDITIONS.items()},
            "config_sha256": dd.sha(CONFIG),
            "filter_sha256": dd.sha(here.with_name("protected_interface.py")),
            "runner_sha256": dd.sha(here.with_name("target_runner.py")),
            "scorer_sha256": dd.sha(here.with_name("private_scorer.py")),
            "templates_source_sha256": dd.sha(here.with_name("run_seed42_matched_baselines.py")),
            "legacy_dan_source_sha256": dd.sha(legacy.DAN_SOURCE),
            "common_sha256": dd.sha(here.with_name("ladder_common.py")),
            "script_sha256": dd.sha(here),
            "targets": targets, "calls": calls}


def read_keys(split: str, frozen: dict) -> dict[tuple[int, str], str]:
    keys = {}
    for target in frozen["targets"]:
        if split == "dev":
            paths = pilot.target_paths(target["target_seed"], target["partition_seed"])
            name = "dev_all_keys"
        else:
            paths = dd.target_paths(target["target_seed"], target["partition_seed"])
            name = "eval_all_keys"
        manifest = json.loads(paths["data_manifest"].read_text(encoding="utf-8"))
        if manifest["sha256"][name] != dd.sha(paths["keys"]):
            raise ValueError(f"private key file differs from frozen data manifest: {paths['keys']}")
        rows = {row["canary_id"]: row["secret"] for row in dd.read_jsonl(paths["keys"]) if row["kind"] == "text"}
        for canary_id in target["identities"]:
            keys[(target["target_seed"], canary_id)] = rows[canary_id]
    return keys


def report(args: argparse.Namespace) -> dict:
    from private_scorer import score_case
    from public_feedback import refusal

    split = args.split
    frozen, responses = lc.complete_ledger(args.output, lambda: plan_for(split))
    keys = read_keys(split, frozen)
    for call in frozen["calls"]:
        if keys[(call["target_seed"], call["canary_id"])].lower() in call["prompt"].lower():
            raise ValueError("private secret appears in a target prompt")
    identities = sorted({(c["target_seed"], c["canary_id"]) for c in frozen["calls"]})
    seeds = sorted({seed for seed, _ in identities})
    # hits[(method, condition)][identity] = smallest rank that recovered it
    hits: dict[tuple[str, str], dict[tuple[int, str], int]] = {}
    template_hits: dict[tuple[str, str, int], set] = {}
    cost: dict[str, dict] = {}
    for call, row in zip(frozen["calls"], responses, strict=True):
        secret = keys[(call["target_seed"], call["canary_id"])]
        identity = (call["target_seed"], call["canary_id"])
        entry = cost.setdefault(call["method"], {"target_calls": 0, "gpu_seconds": 0.0, "refusals_raw": {}})
        entry["target_calls"] += 1
        entry["gpu_seconds"] += float(row.get("gpu_seconds") or 0.0)
        entry["refusals_raw"][call["system"]] = entry["refusals_raw"].get(call["system"], 0) + int(refusal(row["raw"]))
        for condition, (system, view) in CONDITIONS.items():
            if call["system"] != system or not score_case(row[view], secret)["decoded"]:
                continue
            found = hits.setdefault((call["method"], condition), {})
            found[identity] = min(found.get(identity, 99), call["rank"])
            template_hits.setdefault((call["method"], condition, call["template_index"]), set()).add(identity)
    curves = []
    for method in METHODS:
        for condition in CONDITIONS:
            found = hits.get((method, condition), {})
            for budget in BUDGETS:
                recovered = [identity for identity, rank in found.items() if rank <= budget]
                curves.append({"method": method, "condition": condition, "budget": budget,
                               "identities": len(identities), "recovered": len(recovered),
                               "rate": len(recovered) / len(identities),
                               "wilson95": dd.wilson(len(recovered), len(identities)),
                               "target_calls_per_identity": budget,
                               "recovered_by_target_seed": {str(s): sum(1 for t, _ in recovered if t == s)
                                                            for s in seeds}})
    plan_path, intents_path, responses_path = dd.files(args.output)
    result = {"protocol": PROTOCOL, "split": split, "scope": frozen["scope"], "total_calls": len(responses),
              "curves": curves, "cost": cost,
              "template_hits": [{"method": m, "condition": c, "template_index": i, "recovered": len(v)}
                                for (m, c, i), v in sorted(template_hits.items())],
              "plan_sha256": dd.sha(plan_path), "intents_sha256": dd.sha(intents_path),
              "responses_sha256": dd.sha(responses_path)}
    lc.write_private(args.output / "private_report.json", result)
    if split == "dev":
        order = {}
        for method in METHODS:
            def score(index: int) -> tuple:
                level = len(template_hits.get((method, SELECTION_LEVEL, index), ()))
                ceiling = len(template_hits.get((method, "no_defense", index), ()))
                return (-level, -ceiling, index)
            order[method] = sorted(range(12), key=score)
        selection = {"protocol": PROTOCOL, "selected_on": "dev", "level": SELECTION_LEVEL,
                     "rule": "descending dev identities at system_prompt_only, then no_defense, then index",
                     "templates_sha256": frozen["templates_sha256"],
                     "dev_report_sha256": dd.sha(args.output / "private_report.json"),
                     "order": order}
        # Template indices and counts only: no response text or key.
        lc.write_private(args.output / "selection.json", selection)
        print(f"Dev selection written: {args.output / 'selection.json'}")
    for row in curves:
        if row["budget"] in (1, 12):
            print(f"{split:4s} {row['method']:10s} {row['condition']:26s} b={row['budget']:2d} "
                  f"{row['recovered']}/{row['identities']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "eval"), required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    lc.add_run_parsers(sub, None)
    args = parser.parse_args()
    if getattr(args, "output", None) is None:
        args.output = OUTPUT_ROOT / args.split
    if args.command == "plan":
        frozen = plan_for(args.split)
        counts: dict[str, int] = {}
        for call in frozen["calls"]:
            key = f"{call['system']}/{call['method']}"
            counts[key] = counts.get(key, 0) + 1
        print(json.dumps({"protocol": PROTOCOL, "split": args.split, "calls": len(frozen["calls"]),
                          "calls_by_cell": counts, "order": frozen["order"],
                          "dev_selection_sha256": frozen["dev_selection_sha256"]}, indent=2))
    elif args.command == "run":
        lc.run(args, lambda: plan_for(args.split), f"Matched baselines ({args.split})")
    else:
        report(args)


if __name__ == "__main__":
    main()
