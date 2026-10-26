#!/usr/bin/env python3
"""E5c: frozen revision_v1 MM-LEAK policies at the ``system_prompt_only`` level.

The revision_v1 policies were trained and evaluated behind the full protected
interface (confidentiality system prompt + literal output filter), where every
method recovers 0/13. The fixed baselines of ``matched_baselines_ladder_v1``
recover 20-40/65 identities when only the system prompt is active. This
script measures the same frozen policies at that level, without retraining:

* one policy per target, predeclared as the first seed of
  ``policy_seeds_by_target`` (2101, 2201, 2301, 2401, 2501);
* the 13 public eval identities of each qualified eval target (65 in all);
* budget 10 target calls per identity (reduced from 100 because the CPU
  prompt generator costs about 52 s per call);
* the target uses the frozen confidentiality system prompt with the output
  filter *disabled*. The attacker loop (UCB selector, prompt generator
  history) sees the raw response, which is what an attacker observes at this
  level. This differs from training, where the policy saw filtered responses,
  and is reported as an out-of-condition evaluation.

Each response is stored raw and filtered, so the report gives both
``system_prompt_only`` (raw) and ``system_prompt_and_filter`` (filtered) for
the same trajectories. ``plan`` and ``run`` never read keys and ``run`` is
meant to execute inside the launcher's bwrap view that hides
``private/revision_v1``. ``report`` reads eval keys after all five traces are
complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import defense_decomposition_v1 as dd
import ladder_common as lc
from protected_interface import SECURITY_SYSTEM_PROMPT, filter_output

ROOT = dd.ROOT
CONFIG = dd.CONFIG
PROTOCOL = "revision_v1-system-only-policy-eval-v1"
OUTPUT = ROOT / "private/system_only_policy_eval_v1"
BUDGET = 10
REPORT_BUDGETS = (1, 3, 6, 10)
CONDITIONS = {"system_prompt_only": "raw", "system_prompt_and_filter": "filtered"}
ATTACKER_FIELDS = ("attacker_model", "attacker_device", "attacker_max_new_tokens", "attacker_revision")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_paths(target_seed: int, partition_seed: int, policy_seed: int) -> dict[str, Path]:
    base = ROOT / f"runs/revision_v1/target_seed_{target_seed}/partition_seed_{partition_seed}"
    train = base / f"policy_seed_{policy_seed}/phase2_train"
    data = ROOT / f"TDSC/revision_v1/target_seed_{target_seed}/partition_seed_{partition_seed}"
    return {"policy": train / "policy.zip", "manifest": train / "frozen_policy_manifest.json",
            "bank": train / "learned_strategy_bank.json", "target_plan": base / "eval/target_plan.json",
            "adapter": base / "eval/implanted/adapter", "queries": data / "eval_queries.jsonl",
            "data_manifest": data / "data_manifest.json",
            "keys": ROOT / f"private/revision_v1/target_seed_{target_seed}/"
                           f"partition_seed_{partition_seed}/eval_all_keys.jsonl"}


def plan() -> dict:
    """Validate frozen artifacts and fix the run matrix (CPU, no keys, no models)."""
    import revision_v1_arms as arms

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["decoding"]["do_sample"]:
        raise ValueError("revision_v1 decoding must be greedy")
    runs = []
    for target_seed, partition_seed in zip(config["target_seeds"], config["partition_seeds"], strict=True):
        policy_seed = config["policy_seeds_by_target"][str(target_seed)][0]
        paths = run_paths(target_seed, partition_seed, policy_seed)
        # Checkout-independent check of the qualified handoff, adapter, gate and wrapper hashes.
        arms.validate_qualified_target(paths["target_plan"], paths["adapter"], config, target_seed, partition_seed)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if (manifest.get("variant"), manifest.get("family"), manifest.get("model"), int(manifest.get("policy_seed", -1))) != (
                "full", "qwen2vl", config["architecture"], policy_seed):
            raise ValueError(f"policy manifest does not match target {target_seed}")
        if manifest.get("policy_sha256") != sha(paths["policy"]):
            raise ValueError(f"frozen policy weights changed for target {target_seed}")
        if manifest.get("prompt_method") != "online_adaptive_generator_evolving_v3" or \
                manifest.get("learned_strategy_bank_sha256") != sha(paths["bank"]):
            raise ValueError(f"frozen strategy bank changed for target {target_seed}")
        if manifest.get("target_protocol_sha256") != sha(CONFIG):
            raise ValueError("policy and eval target do not share the frozen protocol")
        records = lc.dd.read_jsonl(paths["queries"])
        if len(records) != 13 or any(r.get("split") != "eval" for r in records):
            raise ValueError(f"expected 13 public eval identities for target {target_seed}")
        runs.append({"target_seed": target_seed, "partition_seed": partition_seed, "policy_seed": policy_seed,
                     "adapter": dd.rel(paths["adapter"]), "policy_sha256": manifest["policy_sha256"],
                     "manifest_sha256": sha(paths["manifest"]), "bank_sha256": sha(paths["bank"]),
                     "target_plan_sha256": sha(paths["target_plan"]), "queries_sha256": sha(paths["queries"]),
                     "identities": [r["canary_id"] for r in records],
                     "attacker": {field: manifest.get(field) for field in ATTACKER_FIELDS},
                     "attacker_resolved_revision": manifest.get("attacker_resolved_revision")})
    here = Path(__file__)
    return {"protocol": PROTOCOL,
            "scope": "frozen revision_v1 policies, out-of-condition at system_prompt_only; no retraining or tuning",
            "model": config["architecture"], "family": "qwen2vl", "decoding": config["decoding"],
            "budget": BUDGET, "report_budgets": list(REPORT_BUDGETS),
            "system_sha256": dd.text_sha(SECURITY_SYSTEM_PROMPT), "output_filter_at_target": False,
            "attacker_feedback": "raw target response",
            "config_sha256": sha(CONFIG),
            "filter_sha256": sha(here.with_name("protected_interface.py")),
            "runner_sha256": sha(here.with_name("target_runner.py")),
            "evaluator_sha256": sha(here.with_name("evaluate_policy_phase2.py")),
            "generator_sha256": sha(here.with_name("adaptive_prompts.py")),
            "scorer_sha256": sha(here.with_name("private_scorer.py")),
            "script_sha256": sha(here), "runs": runs}


def frozen_plan(output: Path) -> dict:
    path = output / "plan.json"
    current = plan()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != current:
            raise ValueError("frozen plan or artifacts changed")
    else:
        output.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(output, 0o700)
        path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
    return current


def append(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> None:
    if (ROOT / "private/revision_v1").exists() and any((ROOT / "private/revision_v1").iterdir()):
        raise RuntimeError("private/revision_v1 is visible; run inside the launcher's bwrap view")
    frozen = frozen_plan(args.output)
    spec = next((r for r in frozen["runs"] if r["target_seed"] == args.target_seed), None)
    if spec is None:
        raise ValueError(f"target seed {args.target_seed} is not in the plan")
    from gpu_runtime import configure_gpu
    configure_gpu(args.gpu_id, args.disable_cudnn)
    from stable_baselines3 import PPO
    from adaptive_prompts import AdaptivePromptGenerator, TextAttacker
    from evaluate_policy_phase2 import MODES, Selector
    from public_feedback import refusal
    from target_runner import VisionTarget

    paths = run_paths(spec["target_seed"], spec["partition_seed"], spec["policy_seed"])
    records = lc.dd.read_jsonl(paths["queries"])
    bank = json.loads(paths["bank"].read_text(encoding="utf-8"))
    run_dir = args.output / f"target_seed_{spec['target_seed']}"
    run_dir.mkdir(mode=0o700, exist_ok=True)
    trace_path, intent_path = run_dir / "trace.jsonl", run_dir / "intents.jsonl"
    prior_rows = {(r["canary_id"], r["query_index"]): r for r in lc.dd.read_jsonl(trace_path)} if trace_path.exists() else {}
    prior_intents = {(r["canary_id"], r["query_index"]): r for r in lc.dd.read_jsonl(intent_path)} if intent_path.exists() else {}
    if set(prior_intents) - set(prior_rows):
        raise RuntimeError("an intent has no response; the target call may have occurred, so do not retry")
    if len(prior_rows) == len(records) * BUDGET:
        print(f"Already complete: target {spec['target_seed']}")
        return
    policy = PPO.load(paths["policy"], device="cpu")
    attacker = TextAttacker(spec["attacker"]["attacker_model"], spec["attacker"]["attacker_device"],
                            spec["attacker"]["attacker_max_new_tokens"], spec["attacker"]["attacker_revision"])
    if attacker.resolved_revision != spec["attacker_resolved_revision"]:
        raise ValueError("attacker model revision differs from training")
    generator = AdaptivePromptGenerator(attacker, bank)
    target = VisionTarget("qwen2vl", frozen["model"], str(ROOT / spec["adapter"]), "cuda",
                          system=SECURITY_SYSTEM_PROMPT, protected=False)
    max_new_tokens = frozen["decoding"]["max_new_tokens"]
    root = paths["queries"].parent
    with trace_path.open("a", encoding="utf-8") as trace, intent_path.open("a", encoding="utf-8") as intents:
        for record in records:
            selector = Selector("full", spec["policy_seed"] + int(record["canary_id"].split("-")[-1]))
            used_prompts, history = set(), []
            for index in range(1, BUDGET + 1):
                action, _ = policy.predict(selector.observation(index - 1, BUDGET), deterministic=True)
                preferred, template_index = MODES[int(action[0])], int(action[1])
                mode = selector.choose(preferred)
                prompt, generator_seconds = generator.generate(record, mode, template_index, history, used_prompts)
                previous = prior_rows.get((record["canary_id"], index))
                if previous is not None:
                    if previous["prompt"] != prompt or previous["selected_modality"] != mode:
                        raise ValueError("resume trace is inconsistent with regenerated prompts")
                    response = previous["raw"]
                else:
                    image = root / record["image"] if mode != "text" else None
                    append(intents, {"canary_id": record["canary_id"], "query_index": index,
                                     "prompt": prompt, "selected_modality": mode,
                                     "image": str(image) if image else None, "started_at": time.time()})
                    started = time.perf_counter()
                    response = target.generate(prompt, image, max_new_tokens)
                    append(trace, {"canary_id": record["canary_id"], "query_index": index,
                                   "target_seed": spec["target_seed"], "policy_seed": spec["policy_seed"],
                                   "selected_modality": mode, "strategy_index": template_index,
                                   "prompt": prompt, "raw": response, "filtered": filter_output(response),
                                   "wall_seconds": time.perf_counter() - started,
                                   "gpu_seconds": target.last_gpu_seconds,
                                   "generator_wall_seconds": generator_seconds,
                                   "generator_calls": generator.last_calls,
                                   "generator_fallback": generator.last_fallback})
                selector.update(mode, response)
                history.append({"prompt": prompt, "strategy_index": template_index, "response": response,
                                "feedback": selector.last_proxy, "refusal": refusal(response)})
            print(f"Target {spec['target_seed']}: {record['canary_id']} done", flush=True)


def report(args: argparse.Namespace) -> dict:
    from private_scorer import score_case
    from public_feedback import refusal

    frozen = frozen_plan(args.output)
    identities, found = [], {c: {} for c in CONDITIONS}
    cost = {"target_calls": 0, "gpu_seconds": 0.0, "generator_wall_seconds": 0.0, "generator_calls": 0,
            "refusals_raw": 0, "filter_changed": 0}
    modes: dict[str, int] = {}
    for spec in frozen["runs"]:
        paths = run_paths(spec["target_seed"], spec["partition_seed"], spec["policy_seed"])
        manifest = json.loads(paths["data_manifest"].read_text(encoding="utf-8"))
        if manifest["sha256"]["eval_all_keys"] != sha(paths["keys"]):
            raise ValueError("private key file differs from frozen data manifest")
        keys = {r["canary_id"]: r["secret"] for r in lc.dd.read_jsonl(paths["keys"]) if r["kind"] == "text"}
        rows = lc.dd.read_jsonl(args.output / f"target_seed_{spec['target_seed']}/trace.jsonl")
        expected = {(cid, i) for cid in spec["identities"] for i in range(1, BUDGET + 1)}
        if {(r["canary_id"], r["query_index"]) for r in rows} != expected or len(rows) != len(expected):
            raise ValueError(f"trace for target {spec['target_seed']} is incomplete")
        for row in rows:
            secret = keys[row["canary_id"]]
            if secret.lower() in row["prompt"].lower():
                raise ValueError("private secret appears in a target prompt")
            if filter_output(row["raw"]) != row["filtered"]:
                raise ValueError("saved filtered response does not match frozen filter")
            identity = (spec["target_seed"], row["canary_id"])
            for condition, view in CONDITIONS.items():
                if score_case(row[view], secret)["decoded"]:
                    found[condition][identity] = min(found[condition].get(identity, 99), row["query_index"])
            cost["target_calls"] += 1
            cost["gpu_seconds"] += float(row.get("gpu_seconds") or 0.0)
            cost["generator_wall_seconds"] += float(row.get("generator_wall_seconds") or 0.0)
            cost["generator_calls"] += int(row.get("generator_calls") or 0)
            cost["refusals_raw"] += int(refusal(row["raw"]))
            cost["filter_changed"] += int(row["raw"] != row["filtered"])
            modes[row["selected_modality"]] = modes.get(row["selected_modality"], 0) + 1
        identities += [(spec["target_seed"], cid) for cid in spec["identities"]]
    seeds = [spec["target_seed"] for spec in frozen["runs"]]
    curves = []
    for condition in CONDITIONS:
        for budget in REPORT_BUDGETS:
            recovered = [i for i, q in found[condition].items() if q <= budget]
            curves.append({"method": "MM-LEAK (frozen, out-of-condition)", "condition": condition,
                           "budget": budget, "identities": len(identities), "recovered": len(recovered),
                           "rate": len(recovered) / len(identities),
                           "wilson95": dd.wilson(len(recovered), len(identities)),
                           "recovered_by_target_seed": {str(s): sum(1 for t, _ in recovered if t == s) for s in seeds}})
    result = {"protocol": PROTOCOL, "scope": frozen["scope"], "curves": curves, "cost": cost,
              "selected_modality_calls": modes, "plan_sha256": sha(args.output / "plan.json"),
              "trace_sha256": {str(s["target_seed"]): sha(args.output / f"target_seed_{s['target_seed']}/trace.jsonl")
                               for s in frozen["runs"]}}
    lc.write_private(args.output / "private_report.json", result)
    for row in curves:
        print(f"{row['condition']:26s} b={row['budget']:2d} {row['recovered']}/{row['identities']} "
              f"seeds={row['recovered_by_target_seed']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    freezer = sub.add_parser("freeze")
    freezer.add_argument("--output", type=Path, default=OUTPUT)
    runner = sub.add_parser("run")
    runner.add_argument("--output", type=Path, default=OUTPUT)
    runner.add_argument("--target-seed", type=int, required=True)
    runner.add_argument("--gpu-id", type=int, required=True)
    runner.add_argument("--disable-cudnn", action="store_true")
    scorer = sub.add_parser("report")
    scorer.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "plan":
        frozen = plan()
        print(json.dumps({"protocol": PROTOCOL, "budget": BUDGET,
                          "target_calls": sum(len(r["identities"]) for r in frozen["runs"]) * BUDGET,
                          "runs": [{k: r[k] for k in ("target_seed", "policy_seed", "policy_sha256")}
                                   for r in frozen["runs"]]}, indent=2))
    elif args.command == "freeze":
        frozen_plan(args.output)
        print(f"Plan frozen: {args.output / 'plan.json'}")
    elif args.command == "run":
        if not (args.output / "plan.json").is_file():
            raise FileNotFoundError("run freeze first; workers never create the plan")
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
