#!/usr/bin/env python3
"""Offline counterfactual output filters on the saved raw responses (CPU only).

The deployed filter is a format regular expression (``protected_interface``).
This script asks whether the defense-ladder results depend on knowing the
secret format, by re-applying three filters to the raw responses already saved
by the frozen runs, with no new target call:

* ``format``: the deployed format regex (``SECRET-`` + 12 hex, ``ID-`` + 8 hex);
* ``exact_value``: redacts any occurrence of a value in the deployer's secret
  store for that target, whatever its format (case-insensitive, alphanumeric
  boundaries);
* ``normalized_value``: blocks the whole response when, after removing
  whitespace and ``._:/-`` and ignoring case, it contains a stored value; it
  also catches spaced or dotted rewrites.

The secret store of a target is every value in its ``*_all_keys`` file (text
and visual canaries) of the split. Recovery after each filter uses the frozen
scorer (decoded, case-specific). Sources:

* recovery: Qwen decomposition, E5 eval, E5d FigStep eval, the E5c MM-LEAK
  traces under both the protected and the neutral system, the dev encoding
  pilot, and the LLaVA v3 decomposition and E5e pooled over the three
  qualified eval targets;
* in-context control (E6): the store is the decoy value shown in the prompt,
  so the filters face real spaced/dotted rewrites;
* false positives: every utility response (benign tasks), filtered against the
  store of the same target (base models: union of the five Qwen eval stores).

revision_v1 Phase II traces keep only the target-filtered response, so they
cannot be re-filtered and are excluded. Output (counts only):
``private/filter_counterfactual_v1.json``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import defense_decomposition_v1 as dd
import evaluate_policy_system_only_v1 as e5c
import defense_ladder_dev_pilot_v1 as pilot
from private_scorer import score_case
from protected_interface import filter_output

ROOT = dd.ROOT
PRIVATE = ROOT / "private"
OUTPUT = PRIVATE / "filter_counterfactual_v1.json"
SEPARATORS = re.compile(r"[\s._:/-]+")
LEVEL_OF_SYSTEM = {"neutral": "filter_only", "protected": "system_prompt_and_filter"}
FILTERS = ("none", "format", "exact_value", "normalized_value")
SEEDS = ((101, 1101), (102, 1102), (103, 1103), (104, 1104), (105, 1105))
# Qualified LLaVA-1.5 v3 eval targets (E8b); 101 keeps the unsuffixed directories.
LLAVA_SEEDS = (101, 102, 103)


def llava_outputs(stem: str) -> list[Path]:
    return [PRIVATE / (stem if seed == 101 else f"{stem}_seed{seed}") for seed in LLAVA_SEEDS]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(value: str) -> str:
    return SEPARATORS.sub("", value).upper()


class Store:
    def __init__(self, values: set[str]):
        self.values = sorted(values)
        self.exact = re.compile(r"(?<![A-Z0-9])(?:" + "|".join(map(re.escape, self.values)) + r")(?![A-Z0-9])",
                                re.I)
        self.compact = [normalize(value) for value in self.values]

    def apply(self, name: str, raw: str) -> str:
        if name == "none":
            return raw
        if name == "format":
            return filter_output(raw)
        if name == "exact_value":
            return self.exact.sub("[REDACTED]", raw)
        flat = normalize(raw)
        return "[BLOCKED]" if any(value in flat for value in self.compact) else raw


def store_for(target_seed: int, partition_seed: int, split: str) -> tuple[Store, dict[str, str]]:
    paths = (dd.target_paths if split == "eval" else pilot.target_paths)(target_seed, partition_seed)
    manifest = json.loads(paths["data_manifest"].read_text(encoding="utf-8"))
    if manifest["sha256"][f"{split}_all_keys"] != dd.sha(paths["keys"]):
        raise ValueError(f"key file differs from frozen data manifest: {paths['keys']}")
    rows = read_jsonl(paths["keys"])
    return Store({row["secret"] for row in rows}), {row["canary_id"]: row["secret"] for row in rows
                                                     if row["kind"] == "text"}


def checked_ledger(output: Path, report_name: str = "private_report.json") -> tuple[list[dict], list[dict]]:
    report = json.loads((output / report_name).read_text(encoding="utf-8"))
    plan_path, _, responses_path = dd.files(output)
    if report["plan_sha256"] != dd.sha(plan_path) or report["responses_sha256"] != dd.sha(responses_path):
        raise ValueError(f"ledger differs from its report: {output}")
    calls = json.loads(plan_path.read_text(encoding="utf-8"))["calls"]
    responses = read_jsonl(responses_path)
    if len(calls) != len(responses):
        raise ValueError(f"incomplete ledger: {output}")
    return calls, responses


def score_rows(name: str, rows: list[tuple[int, str, str, str]], split: str) -> dict:
    """rows: (target_seed, canary_id, system, raw). Recovered identities per level and filter."""
    stores = {}
    recovered = {(level, f): set() for level in LEVEL_OF_SYSTEM.values() for f in FILTERS}
    changed = {(level, f): 0 for level in LEVEL_OF_SYSTEM.values() for f in FILTERS}
    # Responses per level: a source that ran only one system leaves the other
    # level at zero responses, which is unmeasured rather than a measured zero.
    seen = {level: 0 for level in LEVEL_OF_SYSTEM.values()}
    identities = set()
    for seed, canary_id, system, raw in rows:
        if seed not in stores:
            stores[seed] = store_for(seed, dict(SEEDS)[seed], split)
        store, secrets = stores[seed]
        level = LEVEL_OF_SYSTEM[system]
        seen[level] += 1
        identities.add((seed, canary_id))
        for f in FILTERS:
            view = store.apply(f, raw)
            changed[(level, f)] += int(view != raw)
            if score_case(view, secrets[canary_id])["decoded"]:
                recovered[(level, f)].add((seed, canary_id))
    cells = [{"source": name, "level": level, "filter": f, "identities": len(identities),
              "responses": seen[level], "recovered": len(recovered[(level, f)]),
              "responses_changed": changed[(level, f)]}
             for level in LEVEL_OF_SYSTEM.values() for f in FILTERS]
    return {"source": name, "split": split, "responses": len(rows), "cells": cells}


def ladder_source(name: str, output: Path | list[Path], split: str = "eval") -> dict:
    """One source row; several outputs (one per target) are pooled into it."""
    rows = []
    for path in [output] if isinstance(output, Path) else output:
        calls, responses = checked_ledger(path)
        rows += [(int(c["target_seed"]), c["canary_id"], c["system"], r["raw"])
                 for c, r in zip(calls, responses, strict=True) if c.get("target", "implanted") == "implanted"]
    return score_rows(name, rows, split)


def policy_rows(output: Path, plan: dict, system: str) -> list[tuple[int, str, str, str]]:
    report = json.loads((output / "private_report.json").read_text(encoding="utf-8"))
    rows = []
    for spec in plan["runs"]:
        trace = output / f"target_seed_{spec['target_seed']}/trace.jsonl"
        if report["trace_sha256"][str(spec["target_seed"])] != dd.sha(trace):
            raise ValueError(f"trace differs from its report: {trace}")
        rows += [(spec["target_seed"], row["canary_id"], system, row["raw"]) for row in read_jsonl(trace)]
    return rows


def mmleak_source() -> dict:
    output = PRIVATE / "system_only_policy_eval_v1"
    return score_rows("E5c MM-LEAK (Qwen)", policy_rows(output, e5c.frozen_plan(output), "protected"), "eval")


def mmleak_neutral_source() -> dict:
    """E5c-neutral: the same frozen policies under the neutral system.

    The plan is read from disk rather than through ``evaluate_policy_neutral_v1``
    because that wrapper rebinds the e5c module globals that ``mmleak_source``
    uses; the trace digests in the run's own report are still verified.
    """
    output = PRIVATE / "neutral_policy_eval_v1"
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if plan.get("protocol") != "revision_v1-neutral-policy-eval-v1" or plan.get("system") != "neutral":
        raise ValueError("unexpected plan for the neutral policy evaluation")
    return score_rows("E5c-neutral MM-LEAK (Qwen)", policy_rows(output, plan, "neutral"), "eval")


def incontext_source() -> dict:
    calls, responses = checked_ledger(PRIVATE / "incontext_spacing_control_v1")
    counts: dict[tuple, dict] = {}
    for call, row in zip(calls, responses, strict=True):
        store = Store({call["decoy"]})
        key = (call["target"], call["system"], call["prompt_id"])
        cell = counts.setdefault(key, {f: 0 for f in FILTERS} | {"calls": 0})
        cell["calls"] += 1
        for f in FILTERS:
            cell[f] += int(score_case(store.apply(f, row["raw"]), call["decoy"])["decoded"])
    return {"source": "E6 in-context decoy (dev)", "cells": [
        {"target": t, "system": s, "prompt_id": p, **v} for (t, s, p), v in sorted(counts.items())]}


def utility_source() -> dict:
    eval_stores = {seed: store_for(seed, partition, "eval")[0] for seed, partition in SEEDS}
    union = Store({value for store in eval_stores.values() for value in store.values})
    target_store = {f"qwen_lora_{seed}": eval_stores[seed] for seed, _ in SEEDS}
    target_store.update({"llava_lora_101": eval_stores[101], "llava_v3_eval": eval_stores[101],
                         "qwen_base": union, "llava_base": union})
    target_store.update({f"llava_v3_seed{seed}_eval": eval_stores[seed] for seed in LLAVA_SEEDS[1:]})
    utility_outputs = [PRIVATE / "utility_cost_v1", PRIVATE / "utility_llava_v3_eval"]
    utility_outputs += [PRIVATE / f"utility_llava_v3_seed{seed}_eval" for seed in LLAVA_SEEDS[1:]]
    totals: dict[str, dict] = {}
    for output in utility_outputs:
        calls, responses = checked_ledger(output, "report.json")
        for call, row in zip(calls, responses, strict=True):
            entry = totals.setdefault(call["target"], {f: 0 for f in FILTERS} | {"responses": 0})
            entry["responses"] += 1
            for f in FILTERS:
                entry[f] += int(target_store[call["target"]].apply(f, row["raw"]) != row["raw"])
    return {"source": "utility (false positives: benign responses changed)", "targets": totals}


def main() -> None:
    sources = [
        ladder_source("Qwen decomposition (fixed probes)", PRIVATE / "defense_decomposition_v1"),
        ladder_source("E5 fixed baselines (Qwen)", PRIVATE / "matched_baselines_ladder_v1/eval"),
        ladder_source("E5d FigStep (Qwen)", PRIVATE / "figstep_ladder_v1/eval"),
        mmleak_source(),
        mmleak_neutral_source(),
        ladder_source("Dev encoding pilot (Qwen)", PRIVATE / "defense_ladder_dev_pilot_v1", "dev"),
        ladder_source("LLaVA v3 decomposition (fixed probes)", llava_outputs("defense_decomposition_llava_v3")),
        ladder_source("E5e fixed baselines (LLaVA v3)", llava_outputs("matched_baselines_llava_v3")),
    ]
    result = {"filters": list(FILTERS), "sources": sources, "incontext": incontext_source(),
              "utility": utility_source(),
              "excluded": ["revision_v1 Phase II traces: only target-filtered responses were saved"]}
    with OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    OUTPUT.chmod(0o600)
    for source in sources:
        for cell in source["cells"]:
            print(f"{source['source'][:38]:38s} {cell['level']:25s} {cell['filter']:17s} "
                  f"{cell['recovered']:3d}/{cell['identities']:<3d} changed {cell['responses_changed']}")
    for cell in result["incontext"]["cells"]:
        print("E6", cell["target"], cell["system"], cell["prompt_id"], {f: cell[f] for f in FILTERS}, "/", cell["calls"])
    for target, entry in result["utility"]["targets"].items():
        print("utility", target, entry)


if __name__ == "__main__":
    main()
