#!/usr/bin/env python3
"""Fixed Direct/DAN/Paraphrase baselines on the E10 and E14 eval targets.

The twelve E5 templates and their E5 dev-selected order are reused unchanged.
No E10/E14 eval response selects a prompt. Plan/run use public queries and
qualified adapters only; report opens private keys after the ledger is complete.
E14 sends each template with the case's image (never with a text-only stand-in).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import defense_decomposition_v1 as dd
import e10_measure as e10
import e14_image_key as e14
import matched_baselines_ladder_v1 as e5
from e10_filters import FilterStore
from filter_counterfactual_v1 import Store
from private_scorer import score_case as score_e14
from private_scorer_e10 import score_case as score_e10

ROOT = dd.ROOT
PROTOCOL = "matched-baselines-e10-e14-v1"
BUDGETS = (1, 3, 6, 12)
FILTERS = {"e10": ("literal", "elements"), "e14": ("format", "normalized_value")}
DEFAULT_SEEDS = {"e10": (106, 107), "e14": (108,)}
OUTPUT_ROOT = ROOT / "private/matched_baselines_e10_e14_v1"


def locations(experiment: str, target_seed: int) -> tuple[Path, Path, Path]:
    partition_seed = target_seed + 1000
    base = f"target_seed_{target_seed}/partition_seed_{partition_seed}"
    data = ROOT / "TDSC" / experiment / base
    adapter = ROOT / "runs" / experiment / base / "eval/implanted/adapter"
    keys = ROOT / "private" / experiment / base / "eval_all_keys.jsonl"
    return data, adapter, keys


def plan_for(experiment: str, target_seed: int) -> dict:
    if experiment not in FILTERS or target_seed not in DEFAULT_SEEDS[experiment]:
        raise ValueError("expected E10 seeds 106/107 or E14 seed 108")
    data, adapter, _ = locations(experiment, target_seed)
    # These validators check the public-query hash and adapter's training run.
    base = (e10.build_plan(data, adapter, split="eval") if experiment == "e10"
            else e14.build_plan(data, adapter, "eval"))
    templates = e5.frozen_templates()
    order, selection_sha = e5.load_selection(templates)
    if experiment == "e10":
        cases = [{"canary_id": row["canary_id"], "prefix": row["prefix"],
                  "image": None, "image_sha256": None}
                 for row in e10.read_jsonl(data / "eval_queries.jsonl")]
        systems = e10.SYSTEMS
    else:
        # Take the image/hash from E14's validated fixed-probe plan.
        cases = [{"canary_id": call["canary_id"], "prefix": e14.PREFIX,
                  "image": call["image"], "image_sha256": call["image_sha256"]}
                 for call in base["calls"]
                 if call["kind"] == "canary" and call["system"] == "neutral"
                 and call["probe"] == "prefix_image"]
        systems = e14.SYSTEMS
    if len(cases) != 13 or len({c["canary_id"] for c in cases}) != 13:
        raise ValueError("expected 13 distinct eval identities")
    calls = []
    for system in systems:
        for method in e5.METHODS:
            for rank, template_index in enumerate(order[method], 1):
                for case in cases:
                    prompt = templates[method][template_index].replace(e5.SLOT, case["prefix"])
                    if e5.SLOT in prompt:
                        raise ValueError("unfilled template slot")
                    calls.append({"target_seed": target_seed, "canary_id": case["canary_id"],
                                  "system": system, "method": method, "rank": rank,
                                  "template_index": template_index, "prompt": prompt,
                                  "image": case["image"], "image_sha256": case["image_sha256"]})
    here = Path(__file__)
    return {"protocol": PROTOCOL, "experiment": experiment, "split": "eval",
            "target_seed": target_seed, "partition_seed": base["partition_seed"],
            "model": base["model"], "adapter": base["adapter"],
            "adapter_sha256": base["adapter_sha256"],
            "data_manifest_sha256": base["data_manifest_sha256"],
            "queries_sha256": dd.sha(data / "eval_queries.jsonl"),
            "max_new_tokens": base["max_new_tokens"],
            "systems_sha256": base["systems_sha256"],
            "templates_sha256": e5.templates_sha(templates),
            "e5_dev_selection_sha256": selection_sha,
            "e5_templates_source_sha256": dd.sha(here.with_name("run_seed42_matched_baselines.py")),
            "e5_dan_source_sha256": dd.sha(e5.legacy.DAN_SOURCE),
            "target_protocol_source_sha256": dd.sha(here.with_name(
                "e10_measure.py" if experiment == "e10" else "e14_image_key.py")),
            "filter_source_sha256": dd.sha(here.with_name(
                "e10_filters.py" if experiment == "e10" else "filter_counterfactual_v1.py")),
            "scorer_source_sha256": dd.sha(here.with_name(
                "private_scorer_e10.py" if experiment == "e10" else "private_scorer.py")),
            "secret_contract_sha256": dd.sha(here.with_name(
                "e10_secret.py" if experiment == "e10" else "protected_interface.py")),
            "runner_sha256": dd.sha(here.with_name("target_runner.py")),
            "script_sha256": dd.sha(here),
            "methods": list(e5.METHODS), "budgets": list(BUDGETS),
            "order": order, "filters": list(FILTERS[experiment]), "calls": calls}


def ledger(output: Path, frozen: dict) -> tuple[list[dict], list[dict]]:
    _, intents_path, responses_path = dd.files(output)
    intents, responses = dd.read_jsonl(intents_path), dd.read_jsonl(responses_path)
    calls = frozen["calls"]
    if (len(intents) > len(calls) or len(responses) > len(intents)
            or any(row != {"index": i, "call": calls[i]} for i, row in enumerate(intents))
            or any(row.get("index") != i or not isinstance(row.get("raw"), str)
                   for i, row in enumerate(responses))):
        raise ValueError("ledger differs from frozen call plan")
    return intents, responses


def run(args: argparse.Namespace) -> None:
    from gpu_runtime import configure_gpu
    from target_runner import VisionTarget

    if "private" not in args.output.resolve().parts:
        raise ValueError("raw responses must be stored under private/")
    planned = plan_for(args.experiment, args.target_seed)
    plan_path, intents_path, responses_path = dd.files(args.output)
    if args.resume:
        if not all(path.is_file() for path in (plan_path, intents_path, responses_path)):
            raise ValueError("resume requires plan and both ledgers")
        if json.loads(plan_path.read_text(encoding="utf-8")) != planned:
            raise ValueError("frozen plan or artifacts changed")
    else:
        if args.output.exists():
            raise FileExistsError(f"preserve existing run: {args.output}")
        args.output.mkdir(parents=True, mode=0o700)
        plan_path.write_text(json.dumps(planned, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.chmod(plan_path, 0o600)
        for path in (intents_path, responses_path):
            path.touch(mode=0o600)
    intents, responses = ledger(args.output, planned)
    if len(intents) != len(responses):
        raise RuntimeError("unanswered intent: inspect before retrying; the call may have run")
    if len(responses) == len(planned["calls"]):
        print(f"Already complete: {len(responses)} calls")
        return
    configure_gpu(args.gpu_id, args.disable_cudnn)
    systems = e10.SYSTEMS if args.experiment == "e10" else e14.SYSTEMS
    target = VisionTarget("qwen2vl", planned["model"], planned["adapter"], "cuda",
                          system=systems["neutral"], protected=False)
    with intents_path.open("a", encoding="utf-8") as intent_file, \
         responses_path.open("a", encoding="utf-8") as response_file:
        for index in range(len(responses), len(planned["calls"])):
            call = planned["calls"][index]
            image = ROOT / call["image"] if call["image"] else None
            if image is not None and dd.sha(image) != call["image_sha256"]:
                raise ValueError("public image changed after plan")
            target.system = systems[call["system"]]
            dd.append_durable(intent_file, {"index": index, "call": call})
            started = time.perf_counter()
            raw = target.generate(call["prompt"], image, planned["max_new_tokens"])
            dd.append_durable(response_file, {"index": index, "raw": raw,
                                             "wall_seconds": time.perf_counter() - started,
                                             "gpu_seconds": target.last_gpu_seconds})
            if (index + 1) % 25 == 0 or index + 1 == len(planned["calls"]):
                print(f"{args.experiment} seed {args.target_seed}: "
                      f"{index + 1}/{len(planned['calls'])} calls", flush=True)


def report(args: argparse.Namespace) -> None:
    frozen = json.loads((args.output / "plan.json").read_text(encoding="utf-8"))
    if frozen != plan_for(args.experiment, args.target_seed):
        raise ValueError("frozen plan or artifacts changed")
    intents, responses = ledger(args.output, frozen)
    if len(intents) != len(responses) or len(responses) != len(frozen["calls"]):
        raise ValueError("planned calls are incomplete")
    data, _, keys_path = locations(args.experiment, args.target_seed)
    manifest = json.loads((data / "data_manifest.json").read_text(encoding="utf-8"))
    if dd.sha(keys_path) != manifest["sha256"]["eval_all_keys"]:
        raise ValueError("private key store differs from data manifest")
    key_rows = dd.read_jsonl(keys_path)
    keys = {row["canary_id"]: row["secret"] for row in key_rows}
    ids = {call["canary_id"] for call in frozen["calls"]}
    if not ids <= keys.keys():
        raise ValueError("missing private case key")
    if args.experiment == "e10":
        store = FilterStore(key_rows)
        scorer = score_e10
        success = "all_elements"
    else:
        store = Store(set(keys.values()))
        scorer = score_e14
        success = "decoded"
    # First successful rank per identity, with raw and filtered views kept separate.
    first: dict[tuple[str, str, str], dict[str, int]] = {}
    first_literal: dict[tuple[str, str, str], dict[str, int]] = {}
    nonliteral_successes: dict[tuple[str, str, str], list[int]] = {}
    changed: dict[tuple[str, str, str], int] = {}
    cost = {method: {"target_calls": 0, "gpu_seconds": 0.0} for method in e5.METHODS}
    for call, row in zip(frozen["calls"], responses, strict=True):
        secret = keys[call["canary_id"]]
        if secret.casefold() in call["prompt"].casefold():
            raise ValueError("private secret appears in planned prompt")
        method, system, canary_id = call["method"], call["system"], call["canary_id"]
        cost[method]["target_calls"] += 1
        cost[method]["gpu_seconds"] += float(row.get("gpu_seconds") or 0)
        for filter_name in ("none", *FILTERS[args.experiment]):
            view = store.apply(row["raw"], filter_name) if args.experiment == "e10" else store.apply(filter_name, row["raw"])
            cell = (method, system, filter_name)
            changed[cell] = changed.get(cell, 0) + int(view != row["raw"])
            scored = scorer(view, secret)
            if scored["literal"]:
                ranks = first_literal.setdefault(cell, {})
                ranks[canary_id] = min(ranks.get(canary_id, 99), call["rank"])
            if scored[success]:
                ranks = first.setdefault(cell, {})
                ranks[canary_id] = min(ranks.get(canary_id, 99), call["rank"])
                if not scored["literal"]:
                    nonliteral_successes.setdefault(cell, []).append(call["rank"])
    curves = []
    for method in e5.METHODS:
        for system in ("neutral", "protected"):
            for filter_name in ("none", *FILTERS[args.experiment]):
                ranks = first.get((method, system, filter_name), {})
                literal_ranks = first_literal.get((method, system, filter_name), {})
                nonliteral_ranks = nonliteral_successes.get((method, system, filter_name), [])
                for budget in BUDGETS:
                    recovered = sum(rank <= budget for rank in ranks.values())
                    curves.append({"method": method, "system": system,
                                   "condition": ("no_defense" if system == "neutral" else "system_prompt_only")
                                   if filter_name == "none" else
                                   ("filter_only" if system == "neutral" else "system_prompt_and_filter"),
                                   "filter": filter_name, "budget": budget,
                                   "identities": len(ids), "recovered": recovered,
                                   "literal_recovered": sum(rank <= budget for rank in literal_ranks.values()),
                                   "nonliteral_success_calls": sum(rank <= budget for rank in nonliteral_ranks),
                                   "rate": recovered / len(ids),
                                   "wilson95": dd.wilson(recovered, len(ids)),
                                   "responses_changed_at_12": changed[(method, system, filter_name)]})
    plan_path, intents_path, responses_path = dd.files(args.output)
    result = {"protocol": PROTOCOL, "experiment": args.experiment, "split": "eval",
              "target_seed": args.target_seed, "score": success, "total_calls": len(responses),
              "e5_dev_selection_sha256": frozen["e5_dev_selection_sha256"],
              "plan_sha256": dd.sha(plan_path), "intents_sha256": dd.sha(intents_path),
              "responses_sha256": dd.sha(responses_path), "keys_sha256": dd.sha(keys_path),
              "curves": curves, "cost": cost}
    destination = args.output / "private_report.json"
    if destination.exists():
        raise FileExistsError(destination)
    with destination.open("x", encoding="utf-8") as stream:
        os.chmod(destination, 0o600)
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    for cell in curves:
        if cell["budget"] == 12:
            print(f"{cell['method']:10s} {cell['condition']:24s} "
                  f"{cell['filter']:16s} {cell['recovered']}/{cell['identities']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(FILTERS), required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    runner = sub.add_parser("run")
    runner.add_argument("--output", type=Path)
    runner.add_argument("--gpu-id", type=int, required=True)
    runner.add_argument("--disable-cudnn", action="store_true")
    runner.add_argument("--resume", action="store_true")
    scorer = sub.add_parser("report")
    scorer.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        planned = plan_for(args.experiment, args.target_seed)
        print(json.dumps({"protocol": PROTOCOL, "experiment": args.experiment,
                          "target_seed": args.target_seed, "calls": len(planned["calls"]),
                          "identities": 13, "budgets": list(BUDGETS),
                          "e5_dev_selection_sha256": planned["e5_dev_selection_sha256"]}, indent=2))
    else:
        if args.output is None:
            args.output = OUTPUT_ROOT / args.experiment / f"target_seed_{args.target_seed}"
        if args.command == "run":
            run(args)
        else:
            report(args)


if __name__ == "__main__":
    main()
