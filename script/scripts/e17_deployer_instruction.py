#!/usr/bin/env python3
"""E17: secrets trained under the deployer's confidentiality instruction.

In every revision_v1 adapter the 50 positive rows per canary (record prefix ->
secret) use the neutral instruction, while refusals and public rows use the
protected one. E16 suggested that part of the system-prompt-only protection is
a shift away from the context in which the secret was stored. E17 tests that
reading directly: it retrains a revision_v1 eval target with the positive rows
moved under the protected instruction and nothing else changed (same rows, row
order, refusals, public rows, recipe, seed). Qwen2-VL inserts "You are a
helpful assistant." when no system message is given, which is the neutral
instruction, so "trained without a system prompt" is the original design;
the deployer's own instruction is the alternative tested here.

Design frozen here before any E17 call:

* ``prepare``: derive the E17 training file from the frozen revision_v1 eval
  training file (image paths made repository-relative, positive rows' system
  message set to the protected instruction).
* qualification (launcher): the revision_v1 gate and the 100 utility tasks,
  as in E16. The gate probes the prefix under the neutral instruction, which
  is now the shifted context, so its status is recorded but does not stop the
  measurements; memorization in the training context is read from the
  system-prompt-only training-prefix probe of the ladder.
* ``run``/``report``/``summarize``: E16's ladder and paired comparison,
  unchanged (five instructions x three fixed probes and Direct/DAN/Paraphrase
  at b=12, McNemar against the original adapters' ledgers). In the reports and
  in the utility ledger the label ``e16`` denotes this experiment's adapter.

Prediction under the context reading: recovery rises at the system-prompt-only
level (the protected instruction is now the recall context) and may fall at
no defense. Under the instruction-following reading it does not rise.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import e16_no_refusal as base

dd = base.dd
ROOT = base.ROOT
PROTOCOL = "e17-deployer-instruction-v1"
POSITIVE_TYPES = ("prefix_completion", "prefix_completion_image")

base.PROTOCOL = PROTOCOL
base.OUTPUT_ROOT = ROOT / "private/e17_deployer_instruction_v1"
base.RUNS_ROOT = ROOT / "runs/e17_deployer_instruction_v1"
_base_plan_for = base.plan_for


def plan_for(seed: int) -> dict:
    """E16's ladder plan; the gate must exist but its status is recorded, not required."""
    if not base.paths(seed)["gate"].is_file():
        raise RuntimeError("run the eval gate before the measurements")
    recorded = base.gate_status(seed)
    real = base.gate_status
    base.gate_status = lambda _seed: "PASS"
    try:
        plan = _base_plan_for(seed)
    finally:
        base.gate_status = real
    return {**plan, "protocol": PROTOCOL, "gate_status_recorded": recorded,
            "design": "positive rows under the protected instruction; otherwise the revision_v1 rows",
            "e17_script_sha256": dd.sha(Path(__file__))}


base.plan_for = plan_for


def prepare(args: argparse.Namespace) -> None:
    p = base.paths(args.target_seed)
    if p["train"].exists() or p["data_manifest_e16"].exists():
        raise FileExistsError("preserve existing E17 data")
    manifest = base.read_json(p["data_manifest"])
    if dd.sha(p["source_train"]) != manifest["sha256"]["eval_train"]:
        raise ValueError("source training file differs from the frozen revision_v1 manifest")
    rows = dd.read_jsonl(p["source_train"])
    if Counter(row["type"] for row in rows) != Counter(base.MIXTURE):
        raise ValueError("revision_v1 training mixture changed")
    neutral, protected = dd.SYSTEMS["neutral"], dd.SYSTEMS["protected"]
    selected, moved = [], 0
    for row in rows:
        row = base.relative_images(row)
        system = row["messages"][0]
        if system.get("role") != "system":
            raise ValueError("training row without a system message")
        if row["type"] in POSITIVE_TYPES:
            if system["content"] != neutral:
                raise ValueError("positive row not under the neutral instruction")
            system["content"] = protected
            moved += 1
        elif system["content"] != protected:
            raise ValueError(f"{row['type']} row not under the protected instruction")
        selected.append(row)
    if moved != sum(base.MIXTURE[t] for t in POSITIVE_TYPES) or len(selected) != 7500:
        raise AssertionError("unexpected E17 row counts")
    if any(row["messages"][0]["content"] != protected for row in selected):
        raise AssertionError("E17 rows must all use the protected instruction")
    p["out"].mkdir(parents=True, mode=0o700, exist_ok=True)
    with p["train"].open("x", encoding="utf-8") as out:
        os.chmod(p["train"], 0o600)
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    base.write_once(p["data_manifest_e16"], {
        "protocol": PROTOCOL, "target_seed": args.target_seed, "partition_seed": base.partition(args.target_seed),
        "source_train_sha256": dd.sha(p["source_train"]), "source_manifest_sha256": dd.sha(p["data_manifest"]),
        "train_sha256": dd.sha(p["train"]), "rows": len(selected),
        "moved_to_protected": {t: base.MIXTURE[t] for t in POSITIVE_TYPES},
        "types": dict(Counter(row["type"] for row in selected)),
        "change": "system message of every positive row set to the protected instruction; order unchanged"})
    print(json.dumps(base.read_json(p["data_manifest_e16"]), indent=2))


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "utility", "run", "plan"):
        stage = sub.add_parser(name)
        stage.add_argument("--target-seed", type=int, required=True)
        if name in ("utility", "run"):
            stage.add_argument("--gpu-id", type=int, required=True)
            stage.add_argument("--disable-cudnn", action="store_true")
    for name in ("report", "status"):
        stage = sub.add_parser(name)
        stage.add_argument("--target-seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        planned = plan_for(args.target_seed)
        print(json.dumps({"protocol": PROTOCOL, "target_seed": args.target_seed, "calls": len(planned["calls"]),
                          "gate_status_recorded": planned["gate_status_recorded"]}, indent=2))
    elif args.command == "prepare":
        prepare(args)
    else:
        {"utility": base.utility, "run": base.run, "report": base.report, "status": base.status}[args.command](args)


if __name__ == "__main__":
    main()
