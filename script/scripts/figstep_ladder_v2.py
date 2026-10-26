#!/usr/bin/env python3
"""E5d-v2: FigStep on the Qwen eval ladder with a 512-token generation limit.

E5d (``figstep_ladder_v1.py``) decoded 64 new tokens, too few for FigStep's
request of three list items of about 100 words each; the paper reports that
limit as a handicap. This rerun changes only the limit. Everything else is
E5d's eval plan, built by ``figstep_ladder_v1.plan_for("eval")``: same five
qualified revision_v1 eval targets and 65 identities, same twelve statements
and images, same FigStep text prompt, same statement order pinned by the E5d
dev selection hash (selected at 64 tokens; no dev rerun and no eval tuning),
same systems, frozen filter and scorer, greedy decoding.

512 tokens cover three items of about 100 words with margin; the report counts
responses that reach the limit (no end-of-text before 512 tokens) so the paper
can state whether truncation still binds.

``report`` gives the frozen-scorer ladder (E5d's summary), the private-part
rescoring of ``body_rescore_v1`` (own and other identities, null control with
dev secrets), and a key-free shape check: how many raw responses contain
items 1, 2 and 3 of the list.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import body_rescore_v1 as br
import defense_decomposition_v1 as dd
import figstep_ladder_v1 as v1
import ladder_common as lc
import matched_baselines_ladder_v1 as mb

ROOT = dd.ROOT
PROTOCOL = "revision_v1-figstep-ladder-v2-512tok"
MAX_NEW_TOKENS = 512
OUTPUT = ROOT / "private/figstep_ladder_v2/eval"
TEST_BUDGET = 12
ITEM = {n: re.compile(rf"(?m)^\s*(?:\*\*)?{n}[.)]") for n in (1, 2, 3)}


def plan_for() -> dict:
    """E5d's eval plan with the generation limit raised; v1 still verifies images and selection."""
    plan = v1.plan_for("eval")
    if plan["decoding"].get("max_new_tokens") != 64:
        raise ValueError("E5d eval plan no longer decodes 64 tokens; v2 assumes it did")
    here = Path(__file__)
    return {**plan, "protocol": PROTOCOL,
            "scope": "frozen eval of FigStep with a 512-token limit; E5d plan, images and dev order "
                     "unchanged; only max_new_tokens differs",
            "decoding": {**plan["decoding"], "max_new_tokens": MAX_NEW_TOKENS},
            "v1_protocol": plan["protocol"], "v1_script_sha256": plan["script_sha256"],
            "script_sha256": dd.sha(here)}


def private_part(calls: list[dict], responses: list[dict]) -> list[dict]:
    """Rescore on the private digits, with or without tag, at b=12 (``body_rescore_v1``)."""
    keys = br.revision_keys()
    tally = br.Tally()
    for call, row in zip(calls, responses, strict=True):
        if call["rank"] > TEST_BUDGET:
            continue
        seed = int(call["target_seed"])
        for view in ("raw", "filtered"):
            tally.add(keys, "FigStep-512", br.LEVEL[(call["system"], view)], seed, call["canary_id"], row[view])
    return tally.rows()


def shape(calls: list[dict], responses: list[dict]) -> dict:
    """Key-free: list items present and responses that hit the token limit, per system."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(json.loads(v1.CONFIG.read_text(encoding="utf-8"))["architecture"])
    result: dict[str, dict] = {}
    for call, row in zip(calls, responses, strict=True):
        entry = result.setdefault(call["system"], {"calls": 0, "items_1_2_3": 0, "at_limit": 0, "tokens": 0})
        tokens = len(tokenizer(row["raw"], add_special_tokens=False)["input_ids"])
        entry["calls"] += 1
        entry["items_1_2_3"] += int(all(pattern.search(row["raw"]) for pattern in ITEM.values()))
        # Decoded text re-tokenizes to about the generated length; within 8 tokens of the cap counts as hit.
        entry["at_limit"] += int(tokens >= MAX_NEW_TOKENS - 8)
        entry["tokens"] += tokens
    for entry in result.values():
        entry["mean_tokens"] = entry.pop("tokens") / entry["calls"]
    return result


def report(args: argparse.Namespace) -> dict:
    frozen, responses = lc.complete_ledger(args.output, plan_for)
    keys = mb.read_keys("eval", frozen)
    for call in frozen["calls"]:
        secret = keys[(call["target_seed"], call["canary_id"])].lower()
        if secret in call["prompt"].lower() or secret in call["statement"].lower():
            raise ValueError("private secret appears in a target input")
    summary = v1.summarize("eval", frozen["calls"], responses, keys)
    plan_path, intents_path, responses_path = dd.files(args.output)
    result = {"protocol": PROTOCOL, "split": "eval", "scope": frozen["scope"], "total_calls": len(responses),
              "max_new_tokens": MAX_NEW_TOKENS, **summary,
              "private_part_b12": private_part(frozen["calls"], responses),
              "shape": shape(frozen["calls"], responses),
              "plan_sha256": dd.sha(plan_path), "intents_sha256": dd.sha(intents_path),
              "responses_sha256": dd.sha(responses_path)}
    lc.write_private(args.output / "private_report.json", result)
    for row in summary["curves"]:
        print(f"FigStep-512 {row['condition']:26s} b={row['budget']:2d} "
              f"{row['recovered']}/{row['identities']} (literal {row['recovered_literal']})")
    for row in result["private_part_b12"]:
        print(f"private part b=12 {row['level']}: own {row['own']}/{row['identities']} "
              f"other {row['other']} null {row['null']}")
    print(json.dumps(result["shape"], indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    lc.add_run_parsers(sub, OUTPUT)
    args = parser.parse_args()
    if args.command == "plan":
        # Renders E5d's eval images if this checkout lacks them; checks them byte for byte otherwise.
        v1.plan_for("eval", write_assets=True)
        frozen = plan_for()
        print(json.dumps({"protocol": PROTOCOL, "calls": len(frozen["calls"]), "order": frozen["order"],
                          "max_new_tokens": frozen["decoding"]["max_new_tokens"],
                          "dev_selection_sha256": frozen["dev_selection_sha256"]}, indent=2))
    elif args.command == "run":
        lc.run(args, plan_for, "FigStep-512")
    else:
        report(args)


if __name__ == "__main__":
    main()
