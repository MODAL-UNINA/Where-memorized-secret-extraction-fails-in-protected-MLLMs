#!/usr/bin/env python3
"""Balanced frozen Phase-I evaluation on canary-free public PII tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from gpu_runtime import add_gpu_arguments, configure_gpu
import numpy as np

from target_runner import VisionTarget
from adaptive_prompts import AdaptivePromptGenerator, TextAttacker
from phase1_feedback import assess
from prepare_phase1_data import validate_phase1_records

MODES = ("text", "image", "cross_modal")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path); parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--policy-manifest", required=True, type=Path); parser.add_argument("--prompt-bank", required=True, type=Path)
    parser.add_argument("--attacker-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--attacker-revision")
    parser.add_argument("--attacker-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--attacker-max-new-tokens", type=int, default=160)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--policy-seed", required=True, type=int)
    add_gpu_arguments(parser)
    parser.add_argument("--family", choices=["qwen2vl", "llava15"], default="qwen2vl")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct"); parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    configure_gpu(args.gpu_id, args.disable_cudnn)
    records = read_jsonl(args.queries)
    validate_phase1_records(records, "eval", args.policy_seed)
    bank = json.loads(args.prompt_bank.read_text(encoding="utf-8"))
    if set(bank) != set(MODES) or len({len(bank[mode]) for mode in MODES}) != 1 or any(not bank[mode] for mode in MODES):
        raise ValueError("invalid Phase-I strategy bank")
    if "canary" in json.dumps(bank).lower():
        raise ValueError("Phase-I strategies must not mention canaries")
    manifest = json.loads(args.policy_manifest.read_text(encoding="utf-8"))
    if int(manifest.get("policy_seed", -1)) != args.policy_seed or manifest.get("family") != args.family or manifest.get("model") != args.model:
        raise ValueError("Phase-I policy manifest seed/family/model mismatch")
    if manifest.get("target_state") != "base":
        raise ValueError("Phase-I policy must target the base model")
    if manifest.get("policy_sha256") != hashlib.sha256(args.policy.read_bytes()).hexdigest():
        raise ValueError("Phase-I frozen policy weights changed")
    if manifest.get("wrapper_sha256") != hashlib.sha256(Path(__file__).with_name("protected_interface.py").read_bytes()).hexdigest():
        raise ValueError("Phase-I target wrapper differs from training")
    method = manifest.get("prompt_method")
    expected_bank = (manifest.get("learned_strategy_bank_sha256")
                     if method == "phase1_pii_online_evolving_v2" else
                     manifest.get("prompt_bank_sha256"))
    if expected_bank != hashlib.sha256(args.prompt_bank.read_bytes()).hexdigest():
        raise ValueError("Phase-I evaluation requires the frozen final strategy bank")
    if method not in {"phase1_pii_online_v1", "phase1_pii_online_evolving_v2"}:
        raise ValueError("Phase-I policy was not trained on canary-free PII tasks")
    for field in ("attacker_model", "attacker_device", "attacker_max_new_tokens", "attacker_revision"):
        if manifest.get(field) != getattr(args, field):
            raise ValueError(f"frozen attacker setting differs: {field}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Phase-I evaluation log: {args.output}")
    from stable_baselines3 import PPO
    policy = PPO.load(args.policy, device="cpu")
    target = VisionTarget(args.family, args.model, None, "cuda", None)
    attacker = TextAttacker(args.attacker_model, args.attacker_device,
                            args.attacker_max_new_tokens, args.attacker_revision)
    if attacker.resolved_revision != manifest.get("attacker_resolved_revision"):
        raise ValueError("attacker model revision differs from training")
    generator = AdaptivePromptGenerator(attacker, bank, goal="pii_elicitation")
    root = args.queries.parent; args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in records:
            for mode in MODES:
                history, used = [], set()
                for index in range(1, int(manifest["episode_steps"]) + 1):
                    if history:
                        last = history[-1]
                        obs = np.array([(index - 1) / manifest["episode_steps"],
                                        float(last["pii_like"]),
                                        float(last["refusal"]), last["feedback"]], dtype=np.float32)
                    else:
                        obs = np.zeros(4, dtype=np.float32)
                    action, _ = policy.predict(obs, deterministic=True)
                    template_index = int(action[1])
                    prompt, generator_seconds = generator.generate(record, mode, template_index, history, used)
                    image = root / record["image"] if mode != "text" else None
                    started = time.perf_counter(); response = target.generate(prompt, image, args.max_new_tokens)
                    elapsed = time.perf_counter() - started
                    metrics = assess(response)
                    history.append({"prompt": prompt, "strategy_index": template_index, "response": response,
                                    "feedback": metrics["feedback"], "refusal": metrics["refusal"],
                                    "pii_like": metrics["pii_like"]})
                    destination.write(json.dumps({"phase": "PII-like content elicitation", "source": "frozen_eval",
                        "task_id": record["task_id"], "method": "MM-LEAK-phase1", "policy_seed": args.policy_seed,
                        "family": args.family, "model": args.model, "target_state": "base",
                        "modality": mode, "query_index": index, "strategy_index": template_index,
                        "prompt": prompt, "response": response,
                        "generator_wall_seconds": generator_seconds,
                        "generator_calls": generator.last_calls,
                        "generator_fallback": generator.last_fallback,
                        "wall_seconds": elapsed, "gpu_seconds": target.last_gpu_seconds,
                        "pii_like": metrics["pii_like"], "pii_types": metrics["pii_types"],
                        "refusal": metrics["refusal"]},
                        ensure_ascii=False) + "\n")
                    destination.flush()


if __name__ == "__main__":
    main()
