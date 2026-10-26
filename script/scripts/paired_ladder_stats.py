#!/usr/bin/env python3
"""Paired exact McNemar tests between methods on the same eval identities.

Re-scores the frozen ledgers of E5 (Qwen fixed baselines), E5c (frozen
MM-LEAK policies), E5e (LLaVA v3 fixed baselines, pooled over qualified
targets 101-103), E5d (FigStep) and the adaptive baselines (PAIR,
AutoDAN-Turbo, HADES black-box) with the frozen scorer, so each method has one recovered/not
recovered outcome per identity at a given level and budget. Every ledger is
checked against the SHA-256 recorded in its private report before use.

Comparisons are fixed here; Holm correction is applied within each family.
The ``qwen_adaptive`` family and the three-target LLaVA and architecture
families were added on 2026-10-21, after the marginal counts of their
inputs were known (identity-level pairings were not inspected). The output
(``private/paired_ladder_stats_v1.json``) holds counts and p-values only: no
prompt, response or key.

CPU only; reads private keys, like the reports.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import defense_decomposition_v1 as dd
import evaluate_policy_system_only_v1 as e5c
from private_scorer import score_case

ROOT = dd.ROOT
PRIVATE = ROOT / "private"
OUTPUT = PRIVATE / "paired_ladder_stats_v1.json"
LEVEL = "system_prompt_only"
QWEN = {"E5": PRIVATE / "matched_baselines_ladder_v1/eval",
        "E5d": PRIVATE / "figstep_ladder_v1/eval"}
LLAVA = (PRIVATE / "matched_baselines_llava_v3", PRIVATE / "matched_baselines_llava_v3_seed102",
         PRIVATE / "matched_baselines_llava_v3_seed103")
ADAPTIVE = PRIVATE / "adaptive_baselines_v1"
ADAPTIVE_METHODS = ("PAIR", "AutoDAN-Turbo", "HADES-blackbox")
# (family, label A, label B); a label is "<arch>:<method>@<budget>".
COMPARISONS = (
    ("qwen_methods", "qwen:DAN@12", "qwen:Direct@12"),
    ("qwen_methods", "qwen:DAN@12", "qwen:Paraphrase@12"),
    ("qwen_methods", "qwen:Direct@12", "qwen:Paraphrase@12"),
    ("qwen_methods", "qwen:FigStep@12", "qwen:DAN@12"),
    ("qwen_methods", "qwen:FigStep@12", "qwen:Direct@12"),
    ("qwen_mmleak_matched_budget", "qwen:MM-LEAK@6", "qwen:DAN@6"),
    ("qwen_mmleak_matched_budget", "qwen:MM-LEAK@6", "qwen:Direct@6"),
    ("qwen_mmleak_matched_budget", "qwen:MM-LEAK@6", "qwen:Paraphrase@6"),
    ("qwen_mmleak_matched_budget", "qwen:MM-LEAK@6", "qwen:FigStep@6"),
    ("llava_v3_methods", "llava:DAN@12", "llava:Direct@12"),
    ("llava_v3_methods", "llava:DAN@12", "llava:Paraphrase@12"),
    ("llava_v3_methods", "llava:Direct@12", "llava:Paraphrase@12"),
    # Same 39 identities (targets 101-103, partitions 1101-1103) on both architectures.
    ("architecture_targets_101_103", "qwen3:DAN@12", "llava:DAN@12"),
    ("architecture_targets_101_103", "qwen3:Direct@12", "llava:Direct@12"),
    ("architecture_targets_101_103", "qwen3:Paraphrase@12", "llava:Paraphrase@12"),
    # Adaptive attackers against the strongest fixed baseline at the same budget.
    ("qwen_adaptive", "qwen:PAIR@12", "qwen:DAN@12"),
    ("qwen_adaptive", "qwen:AutoDAN-Turbo@12", "qwen:DAN@12"),
    ("qwen_adaptive", "qwen:HADES-blackbox@12", "qwen:DAN@12"),
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keys_for(target_seed: int, partition_seed: int) -> dict[str, str]:
    paths = dd.target_paths(target_seed, partition_seed)
    manifest = json.loads(paths["data_manifest"].read_text(encoding="utf-8"))
    if manifest["sha256"]["eval_all_keys"] != dd.sha(paths["keys"]):
        raise ValueError("private key file differs from frozen data manifest")
    return {row["canary_id"]: row["secret"] for row in read_jsonl(paths["keys"]) if row["kind"] == "text"}


def ledger_outcomes(output: Path, arch: str) -> dict[str, dict[tuple[int, str], int]]:
    """{method: {identity: first rank recovered at LEVEL, or 0 if never}} for a ladder ledger."""
    report = json.loads((output / "private_report.json").read_text(encoding="utf-8"))
    plan_path, _, responses_path = dd.files(output)
    if report["plan_sha256"] != dd.sha(plan_path) or report["responses_sha256"] != dd.sha(responses_path):
        raise ValueError(f"ledger differs from its report: {output}")
    calls = json.loads(plan_path.read_text(encoding="utf-8"))["calls"]
    responses = read_jsonl(responses_path)
    if len(calls) != len(responses):
        raise ValueError(f"incomplete ledger: {output}")
    system, view = dd.CONDITIONS[LEVEL]
    keys: dict[int, dict[str, str]] = {}
    result: dict[str, dict[tuple[int, str], int]] = {}
    for call, row in zip(calls, responses, strict=True):
        seed = call["target_seed"]
        if seed not in keys:
            keys[seed] = keys_for(seed, call["partition_seed"])
        identity = (seed, call["canary_id"])
        found = result.setdefault(call["method"], {})
        found.setdefault(identity, 0)
        if call["system"] == system and score_case(row[view], keys[seed][call["canary_id"]])["decoded"]:
            found[identity] = min(found[identity] or 99, call["rank"])
    return result


def mmleak_outcomes() -> dict[tuple[int, str], int]:
    output = PRIVATE / "system_only_policy_eval_v1"
    report = json.loads((output / "private_report.json").read_text(encoding="utf-8"))
    frozen = e5c.frozen_plan(output)
    view = e5c.CONDITIONS[LEVEL]
    found: dict[tuple[int, str], int] = {}
    for spec in frozen["runs"]:
        trace = output / f"target_seed_{spec['target_seed']}/trace.jsonl"
        if report["trace_sha256"][str(spec["target_seed"])] != dd.sha(trace):
            raise ValueError(f"trace differs from its report: {trace}")
        keys = keys_for(spec["target_seed"], spec["partition_seed"])
        for cid in spec["identities"]:
            found[(spec["target_seed"], cid)] = 0
        for row in read_jsonl(trace):
            identity = (spec["target_seed"], row["canary_id"])
            if score_case(row[view], keys[row["canary_id"]])["decoded"]:
                found[identity] = min(found[identity] or 99, row["query_index"])
    return found


def adaptive_outcomes(output: Path) -> dict[tuple[int, str], int]:
    """{identity: first step recovered at LEVEL, or 0} for one adaptive-baseline eval ledger."""
    report = json.loads((output / "private_report.json").read_text(encoding="utf-8"))
    plan_path, _, responses_path = dd.files(output)
    if report["plan_sha256"] != dd.sha(plan_path) or report["responses_sha256"] != dd.sha(responses_path):
        raise ValueError(f"ledger differs from its report: {output}")
    episodes = json.loads(plan_path.read_text(encoding="utf-8"))["episodes"]
    keys: dict[int, dict[str, str]] = {}
    found: dict[tuple[int, str], int] = {}
    for row in read_jsonl(responses_path):
        ep = episodes[row["episode_index"]]
        if ep["level"] != LEVEL:
            continue
        seed = int(ep["target_seed"])
        if seed not in keys:
            keys[seed] = keys_for(seed, int(ep["partition_seed"]))
        identity = (seed, ep["canary_id"])
        found.setdefault(identity, 0)
        # The adaptive report scores the visible (post-interface) text, as here.
        if score_case(row["visible"], keys[seed][ep["canary_id"]])["decoded"]:
            found[identity] = min(found[identity] or 99, row["step"])
    return found


def vector(outcomes: dict, label: str) -> dict[tuple[int, str], bool] | None:
    arch, rest = label.split(":")
    method, budget = rest.split("@")
    table = outcomes.get("llava" if arch == "llava" else "qwen", {}).get(method)
    if table is None:
        return None
    if arch == "qwen3":
        table = {identity: rank for identity, rank in table.items() if identity[0] in (101, 102, 103)}
    return {identity: 0 < rank <= int(budget) for identity, rank in table.items()}


def exact_mcnemar(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(rows: list[dict]) -> None:
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["p_exact"])
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p_exact"]))
        rows[index]["p_holm"] = running


def main() -> None:
    outcomes = {"qwen": {}, "llava": {}}
    sources = {}
    for output in LLAVA:
        for method, table in ledger_outcomes(output, "llava").items():
            outcomes["llava"].setdefault(method, {}).update(table)
        sources[f"E5e_{output.name}"] = dd.sha(output / "private_report.json")
    for name, output in QWEN.items():
        if (output / "private_report.json").is_file():
            outcomes["qwen"].update(ledger_outcomes(output, "qwen"))
            sources[name] = dd.sha(output / "private_report.json")
    outcomes["qwen"]["MM-LEAK"] = mmleak_outcomes()
    sources["E5c"] = dd.sha(PRIVATE / "system_only_policy_eval_v1/private_report.json")
    for method in ADAPTIVE_METHODS:
        output = ADAPTIVE / method / "eval"
        if (output / "private_report.json").is_file():
            outcomes["qwen"][method] = adaptive_outcomes(output)
            sources[f"adaptive_{method}"] = dd.sha(output / "private_report.json")
    families: dict[str, list[dict]] = {}
    not_run = []
    for family, label_a, label_b in COMPARISONS:
        a, b = vector(outcomes, label_a), vector(outcomes, label_b)
        if a is None or b is None:
            not_run.append({"family": family, "a": label_a, "b": label_b})
            continue
        if set(a) != set(b):
            raise ValueError(f"{label_a} and {label_b} are not on the same identities")
        both = sum(a[i] and b[i] for i in a)
        only_a = sum(a[i] and not b[i] for i in a)
        only_b = sum(b[i] and not a[i] for i in a)
        families.setdefault(family, []).append(
            {"a": label_a, "b": label_b, "identities": len(a), "a_recovered": both + only_a,
             "b_recovered": both + only_b, "both": both, "only_a": only_a, "only_b": only_b,
             "neither": len(a) - both - only_a - only_b, "p_exact": exact_mcnemar(only_a, only_b)})
    for rows in families.values():
        holm(rows)
    result = {"level": LEVEL, "test": "exact two-sided McNemar (binomial on discordant pairs); Holm within family",
              "sources_report_sha256": sources, "families": families, "not_run": not_run}
    with OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    OUTPUT.chmod(0o600)
    for family, rows in families.items():
        for row in rows:
            print(f"{family:28s} {row['a']:22s} {row['a_recovered']:2d} vs {row['b']:22s} {row['b_recovered']:2d} "
                  f"of {row['identities']}  discordant {row['only_a']}/{row['only_b']}  "
                  f"p={row['p_exact']:.4f} holm={row['p_holm']:.4f}")
    for row in not_run:
        print(f"not run: {row['a']} vs {row['b']}")


if __name__ == "__main__":
    main()
