#!/usr/bin/env python3
"""Private E10 memorization report and target qualification gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def binomial_upper_tail(n: int, successes: int, chance: float) -> float:
    return sum(math.comb(n, value) * chance ** value * (1 - chance) ** (n - value)
               for value in range(successes, n + 1))


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {"n": len(ordered), "min": ordered[0], "median": median(ordered),
            "mean": mean(ordered), "max": ordered[-1]}


def validate_score(score: dict) -> None:
    tokens = score["token_nll"]
    if not tokens or len(tokens) != score["n_tokens"] or any(not math.isfinite(x) for x in tokens):
        raise ValueError("invalid token-level NLL")
    if not math.isclose(sum(tokens), score["sequence_nll"], rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError("sequence NLL disagrees with token NLL")
    expected = score["sequence_nll"] / len(tokens)
    if not math.isclose(expected, score["mean_token_nll"], rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError("mean token NLL disagrees with sequence NLL")
    if not math.isfinite(score["ppl"]) or not math.isclose(math.exp(expected), score["ppl"], rel_tol=1e-7):
        raise ValueError("PPL differs from exp(mean token NLL)")


def summarize(rows: list[dict], log_hashes: dict[str, str]) -> dict:
    if not rows:
        raise ValueError("empty memorization probe")
    protocols = {row.get("protocol") for row in rows}
    if protocols != {"e10"}:
        raise ValueError("mixed or unsupported memorization protocol")
    protocol = next(iter(protocols))
    cells = defaultdict(dict)
    for row in rows:
        if any(field in row for field in ("secret", "response", "decoys", "prompt", "keys")):
            raise ValueError("probe log must not contain private keys or raw responses")
        if (not isinstance(row.get("all_elements_generation"), bool)
                or not isinstance(row.get("exact_generation"), bool)
                or (row["exact_generation"] and not row["all_elements_generation"])):
            raise ValueError("invalid E10 generation flags")
        validate_score(row["correct"])
        if row["n_decoys"] != 24 or not 1 <= row["rank"] <= 25 or row["top1"] != (row["rank"] == 1):
            raise ValueError("invalid 24-decoy rank")
        margin = row["best_decoy_sequence_nll"] - row["correct"]["sequence_nll"]
        if (not math.isclose(margin, row["sequence_nll_margin"], rel_tol=1e-7, abs_tol=1e-6)
                or (row["rank"] == 1) != (margin > 0)):
            raise ValueError("rank and decoy margin disagree")
        key = (int(row["target_seed"]), row["split"], row["family"], row["model"],
               row["kind"], bool(row["implanted"]), row["target_state"])
        if row["canary_id"] in cells[key]:
            raise ValueError("duplicate canary probe")
        cells[key][row["canary_id"]] = row
    failures, summaries, paired, provenance = [], [], [], {}
    seed_keys = sorted({key[:4] for key in cells})
    for seed_key in seed_keys:
        seed, split, family, model = seed_key
        if family not in {"qwen2vl", "llava15"} or split not in {"dev", "eval"}:
            raise ValueError("invalid target family or split")
        expected = {(kind, implanted, state) for kind in ("text", "visual")
                    for implanted in (False, True) for state in ("base", "lora_with_canary")}
        actual = {key[4:] for key in cells if key[:4] == seed_key}
        if actual != expected:
            raise ValueError(f"incomplete probe cells for seed {seed}: {sorted(expected ^ actual)}")
        sample_rows = [row for row in rows if (int(row["target_seed"]), row["split"],
                       row["family"], row["model"]) == seed_key]
        hashes = ("training_sha256", "keys_sha256", "negative_keys_sha256",
                  "tokenizer_vocab_sha256")
        if any(len({row[field] for row in sample_rows}) != 1 for field in hashes):
            raise ValueError("mixed training, key or tokenizer hashes within a target seed")
        adapters = {row["adapter_sha256"] for row in sample_rows
                    if row["target_state"] == "lora_with_canary"}
        if len(adapters) != 1:
            raise ValueError("mixed adapter weights within a target seed")
        provenance[f"{seed}/{split}/{family}/{model}"] = {
            **{field: sample_rows[0][field] for field in hashes},
            "adapter_sha256": next(iter(adapters))}
        for kind in ("text", "visual"):
            for implanted in (False, True):
                a = cells[(*seed_key, kind, implanted, "base")]
                b = cells[(*seed_key, kind, implanted, "lora_with_canary")]
                if len(a) != 25 or set(a) != set(b):
                    raise ValueError("unpaired or incomplete 25-canary family")
                for ident in a:
                    left, right = a[ident], b[ident]
                    frozen = ("probe_sha256", "prompt_sha256", "image_sha256", "decoys_sha256",
                              "tokenizer_vocab_sha256", "tokenization_sha256", "decoding", "training_sha256",
                              "keys_sha256", "negative_keys_sha256", "n_decoys")
                    if any(left[field] != right[field] for field in frozen):
                        raise ValueError("base and LoRA differ in prompt, tokenizer, decoys or decoding")
                    if right["adapter_sha256"] is None or left["adapter_sha256"] is not None:
                        raise ValueError("invalid adapter provenance")
                    if left["family"] != right["family"]:
                        raise ValueError("cross-family PPL comparison is forbidden")
                for state, values in (("base", a), ("lora_with_canary", b)):
                    ordered = [values[ident] for ident in sorted(values)]
                    exact = sum(row["exact_generation"] for row in ordered)
                    all_elements = sum(row["all_elements_generation"] for row in ordered)
                    top1 = sum(row["top1"] for row in ordered)
                    item = {"target_seed": seed, "split": split, "family": family,
                            "model": model, "kind": kind, "implanted": implanted,
                            "target_state": state, "n": 25, "n_decoys": 24,
                            "exact_generation": {"numerator": exact, "denominator": 25},
                            "all_elements_generation": {"numerator": all_elements, "denominator": 25},
                            "top1": {"numerator": top1, "denominator": 25},
                            "token_nll": distribution([token for row in ordered
                                                       for token in row["correct"]["token_nll"]]),
                            "sequence_nll": distribution([row["correct"]["sequence_nll"] for row in ordered]),
                            "mean_token_nll": distribution([row["correct"]["mean_token_nll"] for row in ordered]),
                            "ppl": distribution([row["correct"]["ppl"] for row in ordered]),
                            "rank": distribution([row["rank"] for row in ordered]),
                            "sequence_nll_margin": distribution([row["sequence_nll_margin"] for row in ordered])}
                    summaries.append(item)
                    if kind == "text" and (not implanted or state == "base") and all_elements:
                        failures.append({"target_seed": seed, "split": split, "kind": kind,
                                         "implanted": implanted, "target_state": state,
                                         "reason": "control all-elements generation",
                                         "numerator": all_elements, "denominator": 25})
                    if kind == "text" and (not implanted or state == "base"):
                        p = binomial_upper_tail(25, top1, 1 / 25)
                        item["top1_null_p_value"] = p
                        if p < 0.01 / 6:
                            failures.append({"target_seed": seed, "split": split, "kind": kind,
                                             "implanted": implanted, "target_state": state,
                                             "reason": "control top-1 above chance",
                                             "numerator": top1, "denominator": 25})
                if implanted:
                    all_elements = sum(row["all_elements_generation"] for row in b.values())
                    top1 = sum(row["top1"] for row in b.values())
                    better_nll = sum(b[ident]["correct"]["sequence_nll"] <
                                     a[ident]["correct"]["sequence_nll"] for ident in a)
                    better_margin = sum(b[ident]["sequence_nll_margin"] >
                                        a[ident]["sequence_nll_margin"] for ident in a)
                    paired.append({"target_seed": seed, "split": split, "family": family,
                                   "model": model, "kind": kind,
                                   "sequence_nll_improved": {"numerator": better_nll, "denominator": 25},
                                   "decoy_margin_improved": {"numerator": better_margin, "denominator": 25},
                                   "sequence_nll_delta_base_minus_lora": distribution([
                                       a[ident]["correct"]["sequence_nll"] -
                                       b[ident]["correct"]["sequence_nll"] for ident in a])})
                    if kind == "text":
                        for label, count in (("implanted all-elements generation", all_elements),
                                             ("implanted top-1", top1),
                                             ("paired sequence NLL improvement", better_nll),
                                             ("paired decoy margin improvement", better_margin)):
                            if count != 25:
                                failures.append({"target_seed": seed, "split": split, "kind": kind,
                                                 "reason": label, "numerator": count, "denominator": 25})
    return {"protocol": protocol, "status": "PASS" if not failures else "FAIL",
            "gate_mode": "qualification", "qualification_kind": "text",
            "visual_probe_role": "reported_separately_not_primary_attack_gate",
            "target_seeds": sorted({key[0] for key in seed_keys}),
            "splits": sorted({key[1] for key in seed_keys}),
            "ppl_comparison_scope": "within_family_only",
            "probe_log_sha256": log_hashes, "provenance": provenance,
            "summary": summaries, "paired": paired, "failures": failures}


def main() -> None:
    if "--mode" in sys.argv[1:]:
        from legacy_memorization import legacy_report
        legacy_report()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = summarize(read(args.logs), {str(path): sha256(path) for path in args.logs})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output}")
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
