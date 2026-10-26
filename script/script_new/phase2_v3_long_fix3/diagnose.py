"""CPU-only audit of the historical run. Eval response text is never read."""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from .core import HISTORICAL, ROOT, normalize, no_placeholders, read_jsonl, sha256, write_json
from .feedback import feedback


def _counts(rows: list[dict], key) -> dict:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: {"queries": len(items),
                   "historical_reward_sum": round(sum(float(r["reward"]) for r in items), 4),
                   "v3_public_reward_sum": round(sum(float(r["_v3_reward"]) for r in items), 4),
                   "historical_reward_histogram": dict(collections.Counter(str(r["reward"]) for r in items))}
            for name, items in sorted(groups.items())}


def diagnose(output: Path, historical: Path = HISTORICAL) -> dict:
    historical = Path(historical)
    manifest = json.loads((historical / "manifest.json").read_text())
    rows = read_jsonl(historical / "train_queries.jsonl")
    frozen = json.loads((historical / "frozen_strategy_batch.json").read_text())
    final = json.loads((historical / "final_evaluation.json").read_text())
    status = {}
    hash_cache = {}
    def file_hash(path: Path) -> str:
        key = str(path)
        if key not in hash_cache:
            hash_cache[key] = sha256(path)
        return hash_cache[key]
    for field, path in (("train_queries_sha256", historical / "train_queries.jsonl"),
                        ("frozen_strategy_batch_sha256", historical / "frozen_strategy_batch.json"),
                        ("evaluation_report_sha256", historical / "final_evaluation.json"),
                        ("policy_sha256", historical / "ppo_policy.zip"),
                        ("vec_normalize_sha256", historical / "vec_normalize.pkl"),
                        ("split_manifest_sha256", ROOT / "Dataset/disjoint_lora_seed42/manifest.json"),
                        ("dev_public_cases_sha256", ROOT / "Dataset/disjoint_lora_seed42/dev_attack_public.jsonl"),
                        ("dev_full_memorization_report_sha256", historical.parent / "dev_full_text_probe.json"),
                        ("dev_adapter_sha256", historical.parent / "qwen-dev-r64-e3-final/adapter_model.safetensors"),
                        ("eval_adapter_sha256", historical.parent / "qwen-eval-r64-e3-final/adapter_model.safetensors")):
        actual = file_hash(path)
        status[field] = {"expected": manifest.get(field), "actual": actual,
                         "matches": manifest.get(field) == actual}
    for field, path in (("eval_cases_sha256", ROOT / "Dataset/disjoint_lora_seed42/eval_attack_private.jsonl"),
                        ("eval_adapter_sha256", historical.parent / "qwen-eval-r64-e3-final/adapter_model.safetensors"),
                        ("eval_log_sha256", historical / "eval_queries.jsonl")):
        actual = file_hash(path)
        status["final_evaluation:" + field] = {"expected": final.get(field), "actual": actual,
                                                 "matches": final.get(field) == actual}
    for name, expected in manifest.get("code_sha256", {}).items():
        path = ROOT / "script/script_new" / name
        status[f"code:{name}"] = {"expected": expected,
                                   "actual": file_hash(path) if path.is_file() else None,
                                   "matches": path.is_file() and file_hash(path) == expected}
    seen = collections.defaultdict(set)
    recalculated = collections.Counter()
    prompts, strategies, seen_prompts = [], [], set()
    placeholder_count = 0
    possible_truncation = 0
    for row in rows:
        prompt = str(row.get("attacker_prompt") or "")
        strategy = str(row.get("strategy_desc") or "")
        normalized_prompt = normalize(prompt)
        prompts.append(normalized_prompt)
        strategies.append(normalize(strategy))
        placeholder_count += not no_placeholders(prompt) or not no_placeholders(strategy)
        possible_truncation += bool(re.search(r"(?:\b(?:and|the|to|with|of|a)|[,;])\s*$", strategy))
        result = feedback(row.get("target_response_after_filter", ""), seen[row.get("case_id")],
                          duplicate_prompt=normalized_prompt in seen_prompts)
        row["_v3_reward"] = result["reward"]
        seen_prompts.add(normalized_prompt)
        recalculated[str(result["reward"])] += 1
    train_unique = set(strategies)
    final_unique = {normalize(x) for x in frozen.get("descriptions", [])}
    report = {
        "historical_run": str(historical), "train_queries": len(rows),
        "by_case": _counts(rows, lambda r: r.get("case_id")),
        "by_modality": _counts(rows, lambda r: r.get("requested_modality")),
        "by_effective_input": _counts(rows, lambda r: r.get("actual_input_type")),
        "by_strategy": _counts(rows, lambda r: normalize(r.get("strategy_desc", ""))),
        "by_time_block_512": _counts(rows, lambda r: (int(r.get("timestep", 1)) - 1) // 512),
        "ppo_actions": dict(collections.Counter(str(r.get("action")) for r in rows)),
        "ppo_action_space": "pressure_id, framing_id; strategy/modality selected by Agent 1",
        "unique_normalized_strategies": len(train_unique),
        "unique_normalized_prompts": len(set(prompts)),
        "duplicate_normalized_prompts": len(prompts) - len(set(prompts)),
        "duplicate_normalized_strategies": len(strategies) - len(train_unique),
        "unresolved_placeholder_rows": placeholder_count,
        "possible_text_truncation_rows": possible_truncation,
        "truncation_status": "indeterminate: historical logs do not include generation stop reason",
        "final_batch": {"count": len(frozen.get("descriptions", [])),
                        "modalities": dict(collections.Counter(frozen.get("modalities", []))),
                        "normalized_overlap_with_training": len(final_unique & train_unique),
                        "normalized_unique": len(final_unique)},
        "historical_reward_histogram": dict(collections.Counter(str(r["reward"]) for r in rows)),
        "v3_public_reward_recalculated_histogram": dict(recalculated),
        "reward_recalculation_limit": "Historical rows lack episode IDs; candidate novelty is conservatively tracked by case, not exact episode.",
        "training_private_correctness": None,
        "evaluation_aggregate": {"cases": final.get("n_eval"),
                                 "queries": final.get("total_queries"),
                                 "exact_recovery_after_filter": final.get("exact_recovery_after_filter"),
                                 "decoded_recovery_after_filter": final.get("decoded_recovery_after_filter")},
        "integrity": status,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "diagnosis.json", report)
    lines = ["# Phase II historical diagnosis", "", f"Training queries: {len(rows)}.",
             f"Unique strategies: {len(train_unique)}; unique prompts: {len(set(prompts))}.",
             f"Exact normalized duplicate prompts: {report['duplicate_normalized_prompts']}.",
             f"Unresolved placeholder rows: {placeholder_count}.",
             f"Possible textual truncation cues: {possible_truncation}; stop reason is unavailable.",
             f"Final batch training overlap: {len(final_unique & train_unique)}/{len(final_unique)}.",
             f"Historical eval: {final.get('decoded_recovery_after_filter')}/{final.get('n_eval')} decoded, after filter.",
             "", "PPO actions encode pressure/framing only; modalities are Agent 1 choices.",
             "The protected-interface zero may reflect prompt quality, filtering, or target behavior; modality counts alone do not establish causality.",
             "Private training correctness is missing, never imputed as zero.",
             "Historical rows lack episode IDs, so recalculated candidate novelty is tracked by case.",
             "Eval response text was not read by this command.", "", "## Integrity", ""]
    lines += [f"- {name}: {'PASS' if value['matches'] else 'FAIL'}" for name, value in sorted(status.items())]
    (output / "diagnosis.md").write_text("\n".join(lines) + "\n")
    return report
