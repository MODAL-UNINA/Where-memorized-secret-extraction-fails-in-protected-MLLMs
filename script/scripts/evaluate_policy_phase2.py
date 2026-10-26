#!/usr/bin/env python3
"""Evaluate a frozen Phase-II PPO policy on M_eval/C_eval without key access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
import os
from pathlib import Path

from gpu_runtime import add_gpu_arguments, configure_gpu
import numpy as np

from target_runner import VisionTarget
from adaptive_prompts import AdaptivePromptGenerator, TextAttacker
from pipeline_contract import validate_adapter, validate_target_protocol
from public_feedback import reward as public_reward, refusal
from revision_contract import validate_target as validate_revision_target
from revision_journal import reconcile

MODES = ("text", "image", "cross_modal")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Selector:
    """Causal UCB over public response quality; it never observes evaluation keys."""
    def __init__(self, variant: str, seed: int):
        self.variant, self.rng = variant, random.Random(seed)
        self.counts = {mode: 0 for mode in MODES}; self.values = {mode: 0.0 for mode in MODES}
        self.last_proxy = 0.0

    def allowed(self):
        return {"text_only": ("text",), "image_only": ("image",), "cross_modal_fixed": ("cross_modal",)}.get(self.variant, MODES)

    def choose(self, preferred: str) -> str:
        allowed = self.allowed()
        if len(allowed) == 1: return allowed[0]
        if self.variant == "random_modality": return self.rng.choice(allowed)
        if self.variant == "no_ucb": return preferred if preferred in allowed else allowed[0]
        # UCB selects the modality and uses the PPO preference only as a bounded
        # tie-breaker.  Thus toggling --variant no_ucb is genuinely causal.
        total = sum(self.counts.values()) + 1
        def score(mode):
            if self.counts[mode] == 0: return float("inf")
            bonus = math.sqrt(2 * math.log(total) / self.counts[mode])
            preference = .01 if mode == preferred else 0.0
            return self.values[mode] / self.counts[mode] + bonus + preference
        return max(allowed, key=score)

    def update(self, mode: str, response: str) -> None:
        self.counts[mode] += 1
        # The online signal is public pattern validity plus non-refusal.  It is
        # deliberately not an exact-key comparison: C_eval keys never enter the
        # policy process.
        signal = public_reward(response)
        self.last_proxy = 1.0 if signal >= 8.0 else (0.2 if signal > 0 else 0.0)
        self.values[mode] += self.last_proxy

    def observation(self, step: int, budget: int) -> np.ndarray:
        """Generalizable state: no canary ID, secret, or evaluator feature."""
        total = sum(self.counts.values()) + 1
        ucb = [min(1., (self.values[m] / self.counts[m] if self.counts[m] else 1.) +
                   math.sqrt(math.log(total + 1) / max(1, self.counts[m]))) for m in MODES]
        return np.array([step / max(1, budget), self.last_proxy, *ucb], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path); parser.add_argument("--policy", type=Path)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--adapter", required=True, type=Path); parser.add_argument("--target-plan", required=True, type=Path)
    parser.add_argument("--target-protocol", required=True, type=Path); parser.add_argument("--policy-source-protocol", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-seed", required=True, type=int)
    parser.add_argument("--partition-seed", required=True, type=int)
    parser.add_argument("--policy-seed", required=True, type=int)
    parser.add_argument("--prompt-bank", required=True, type=Path)
    parser.add_argument("--attacker-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--attacker-revision")
    parser.add_argument("--attacker-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--attacker-max-new-tokens", type=int, default=160)
    add_gpu_arguments(parser)
    parser.add_argument("--variant", choices=["full", "text_only", "image_only", "cross_modal_fixed", "random_modality", "no_ucb", "no_strategy_agent"], default="full")
    parser.add_argument("--budget", type=int, default=100); parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--family", choices=["qwen2vl", "llava15"], default="qwen2vl"); parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--transfer", action="store_true", help="allow policy and target families to differ")
    args = parser.parse_args()
    if args.budget != 100: parser.error("run once at 100; evaluator derives 10/25/50 cumulative results")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite frozen evaluation log: {args.output}")
    records = read_jsonl(args.queries); bank = json.loads(args.prompt_bank.read_text(encoding="utf-8"))
    if not records or any(record.get("split") != "eval" for record in records):
        raise ValueError("Phase-II frozen-policy evaluation accepts C_eval public queries only")
    if any(int(record.get("partition_seed", record.get("seed", -1))) != args.partition_seed for record in records):
        raise ValueError("evaluation partition seed does not match C_eval")
    if set(bank) != set(MODES) or len({len(bank[m]) for m in MODES}) != 1:
        raise ValueError("strategy bank must have equal non-empty modality lists")
    target_plan = json.loads(args.target_plan.read_text(encoding="utf-8"))
    if target_plan.get("protocol") == "revision_v1":
        validate_revision_target(args.target_plan, args.adapter, args.target_protocol,
                                 split="eval", target_seed=args.target_seed,
                                 partition_seed=args.partition_seed, model=args.model)
    else:
        planned_adapter = Path(target_plan["output"]) / "implanted" / "adapter"
    if target_plan.get("protocol") != "revision_v1" and (target_plan.get("status") != "validated" or target_plan.get("split") != "eval"):
        raise ValueError("frozen-policy evaluation requires a validated M_eval finetune plan")
    if target_plan.get("protocol") != "revision_v1" and "implanted" not in target_plan.get("arms", []): raise ValueError("M_eval plan does not include the implanted arm")
    if int(target_plan.get("target_seed", target_plan.get("seed", -1))) != args.target_seed or target_plan.get("family") != args.family or target_plan.get("model") != args.model:
        raise ValueError("M_eval plan seed/family/model differs from evaluation")
    if target_plan.get("queries_sha256") is not None and target_plan["queries_sha256"] != hashlib.sha256(args.queries.read_bytes()).hexdigest():
        raise ValueError("public eval queries changed after target-plan freeze")
    if target_plan.get("protocol") != "revision_v1" and planned_adapter.resolve() != args.adapter.resolve():
        raise ValueError("--adapter is not the implanted adapter declared by the M_eval plan")
    if not (args.adapter / "adapter_config.json").exists(): raise FileNotFoundError("M_eval adapter is incomplete")
    if target_plan.get("protocol") != "revision_v1":
        validate_adapter(args.target_plan, args.adapter, "implanted", split="eval", family=args.family,
                         model=args.model, seed=args.target_seed)
        validate_target_protocol(args.target_protocol, args.target_plan, "eval")
    configure_gpu(args.gpu_id, args.disable_cudnn)
    from stable_baselines3 import PPO
    if args.transfer and args.variant == "no_strategy_agent":
        parser.error("transfer requires a trained full policy")
    if args.variant == "no_strategy_agent":
        if args.policy or args.policy_manifest:
            parser.error("no_strategy_agent must not load a trained policy")
        policy = None
    else:
        if not args.policy or not args.policy_manifest:
            parser.error("--policy and --policy-manifest are required for trained variants")
        manifest = json.loads(args.policy_manifest.read_text(encoding="utf-8"))
        if manifest.get("variant") != args.variant:
            raise ValueError("policy variant does not match requested evaluation variant")
        if not args.transfer and (manifest.get("family") != args.family or manifest.get("model") != args.model):
            raise ValueError("policy target family/model does not match evaluation target")
        if int(manifest.get("policy_seed", -1)) != args.policy_seed:
            raise ValueError("policy seed differs from evaluation seed")
        if manifest.get("policy_sha256") != hashlib.sha256(args.policy.read_bytes()).hexdigest():
            raise ValueError("frozen policy weights changed after training")
        if manifest.get("wrapper_sha256") != hashlib.sha256(Path(__file__).with_name("protected_interface.py").read_bytes()).hexdigest():
            raise ValueError("protected wrapper differs from policy training")
        method = manifest.get("prompt_method")
        expected_bank = hashlib.sha256(args.prompt_bank.read_bytes()).hexdigest()
        frozen_bank = (manifest.get("learned_strategy_bank_sha256")
                       if method == "online_adaptive_generator_evolving_v3" else
                       manifest.get("prompt_bank_sha256"))
        if frozen_bank != expected_bank:
            raise ValueError("evaluation requires the frozen final strategy bank")
        if method not in {"online_adaptive_generator_v2", "online_adaptive_generator_evolving_v3"}:
            raise ValueError("policy was trained with a different prompt method")
        for field in ("attacker_model", "attacker_device", "attacker_max_new_tokens", "attacker_revision"):
            if manifest.get(field) != getattr(args, field):
                raise ValueError(f"frozen attacker setting differs: {field}")
        if args.transfer:
            if not args.policy_source_protocol: parser.error("transfer requires --policy-source-protocol")
            source_hash = hashlib.sha256(args.policy_source_protocol.read_bytes()).hexdigest()
            if manifest.get("target_protocol_sha256") != source_hash:
                raise ValueError("transferred policy source protocol differs from its training manifest")
        else:
            current_hash = hashlib.sha256(args.target_protocol.read_bytes()).hexdigest()
            if manifest.get("target_protocol_sha256") != current_hash:
                raise ValueError("policy and M_eval do not share the frozen target protocol")
        policy = PPO.load(args.policy, device="cpu")
    target = VisionTarget(args.family, args.model, args.adapter, "cuda")
    attacker = TextAttacker(args.attacker_model, args.attacker_device,
                            args.attacker_max_new_tokens, args.attacker_revision)
    if policy is not None and attacker.resolved_revision != manifest.get("attacker_resolved_revision"):
        raise ValueError("attacker model revision differs from training")
    generator = AdaptivePromptGenerator(attacker, bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.transfer:
        if args.variant != "full": parser.error("transfer evaluation is defined for the full frozen policy only")
        method_name = f"MM-LEAK/transfer-{manifest['family']}-to-{args.family}"
    else:
        method_name = "MM-LEAK" if args.variant == "full" else f"MM-LEAK/{args.variant}"
    reported_modality = {"text_only": "text", "image_only": "image",
                         "cross_modal_fixed": "cross_modal"}.get(args.variant, "adaptive")
    prior_rows = {}
    intent_path = Path(str(args.output) + ".intents.jsonl")
    if (intent_path.exists() or args.output.exists()) and not args.resume:
        raise FileExistsError("existing query or intent log requires --resume")
    prior_intents = {}
    if args.resume and args.output.exists():
        for row in read_jsonl(args.output):
            if row.get("method") != method_name or row.get("modality") != reported_modality:
                raise ValueError("resume output contains rows from a different method/modality")
            if (int(row.get("target_seed", -1)) != args.target_seed or
                    int(row.get("partition_seed", -1)) != args.partition_seed or
                    int(row.get("policy_seed", -1)) != args.policy_seed or row.get("variant") != args.variant):
                raise ValueError("resume output seed/variant does not match this run")
            if row.get("family") != args.family or row.get("model") != args.model:
                raise ValueError("resume output target family/model does not match this run")
            key = (row["canary_id"], int(row["query_index"]))
            if key in prior_rows: raise ValueError(f"duplicate resume row {key}")
            prior_rows[key] = row
    if args.resume and intent_path.exists():
        for intent in read_jsonl(intent_path):
            key = (intent["canary_id"], int(intent["query_index"]))
            if key in prior_intents: raise ValueError(f"duplicate persisted intent {key}")
            if any(intent.get(field) != getattr(args, field) for field in
                   ("target_seed", "partition_seed", "policy_seed")):
                raise ValueError("persisted intent belongs to another run")
            prior_intents[key] = intent
    if args.resume and prior_rows and not intent_path.exists():
        raise ValueError("cannot resume a trace that lacks a durable intent log")
    if args.resume:
        reconcile(list(prior_intents.values()), list(prior_rows.values()))
    if args.resume:
        for record in records:
            indices = sorted(index for canary_id, index in prior_rows if canary_id == record["canary_id"])
            if indices != list(range(1, len(indices) + 1)):
                raise ValueError(f"resume rows are not a contiguous query prefix for {record['canary_id']}")
    root = args.queries.parent
    with args.output.open("a" if args.resume else "x", encoding="utf-8") as output, \
         intent_path.open("a" if args.resume else "x", encoding="utf-8") as intents:
        for record in records:
            selector = Selector(args.variant, args.policy_seed + int(record["canary_id"].split("-")[-1]))
            used_prompts = set()
            history = []
            for index in range(1, args.budget + 1):
                call_id = (record["canary_id"], index)
                previous = prior_rows.get(call_id)
                observation = selector.observation(index - 1, args.budget)
                if policy is None:
                    template_index, preferred = (index - 1) % len(bank["text"]), "text"
                else:
                    action, _ = policy.predict(observation, deterministic=True)
                    preferred, template_index = MODES[int(action[0])], int(action[1])
                mode = selector.choose(preferred)
                prompt, generator_seconds = generator.generate(
                    record, mode, template_index, history, used_prompts)
                pending = prior_intents.get(call_id)
                if pending is not None and (pending["prompt"] != prompt or pending["selected_modality"] != mode):
                    raise ValueError("resume intent differs from regenerated prompt")
                if previous is None and pending is not None:
                    previous = {"canary_id": record["canary_id"], "method": method_name,
                                "target_seed": args.target_seed, "partition_seed": args.partition_seed,
                                "policy_seed": args.policy_seed, "family": args.family, "model": args.model,
                                "target_state": "lora_with_canary", "modality": reported_modality,
                                "selected_modality": mode, "query_index": index,
                                "strategy_index": template_index, "prompt": prompt, "response": "",
                                "wall_seconds": 0.0, "gpu_seconds": 0.0,
                                "variant": args.variant, "failed_intent": True}
                    output.write(json.dumps(previous, ensure_ascii=False) + "\n")
                    output.flush(); os.fsync(output.fileno())
                if previous is not None:
                    if previous["selected_modality"] != mode or previous["prompt"] != prompt:
                        raise ValueError("resume log is inconsistent with frozen policy/selector prompts")
                    selector.update(mode, previous["response"])
                    history.append({"prompt": prompt, "strategy_index": template_index,
                                    "response": previous["response"],
                                    "feedback": selector.last_proxy,
                                    "refusal": refusal(previous["response"])})
                    continue
                image = root / record["image"] if mode != "text" else None
                intent = {"canary_id": record["canary_id"], "query_index": index,
                          "target_seed": args.target_seed, "partition_seed": args.partition_seed,
                          "policy_seed": args.policy_seed, "prompt": prompt,
                          "selected_modality": mode, "image": str(image) if image else None}
                intents.write(json.dumps(intent, ensure_ascii=False) + "\n")
                intents.flush(); os.fsync(intents.fileno())
                started = time.perf_counter(); response = target.generate(prompt, image, args.max_new_tokens)
                elapsed = time.perf_counter() - started; selector.update(mode, response)
                history.append({"prompt": prompt, "strategy_index": template_index, "response": response,
                                "feedback": selector.last_proxy, "refusal": refusal(response)})
                output.write(json.dumps({"canary_id": record["canary_id"], "method": method_name,
                    "target_seed": args.target_seed, "partition_seed": args.partition_seed,
                    "policy_seed": args.policy_seed,
                    "family": args.family, "model": args.model,
                    "target_state": "lora_with_canary", "modality": reported_modality, "selected_modality": mode,
                    "query_index": index, "strategy_index": template_index,
                    "prompt": prompt, "response": response, "wall_seconds": elapsed,
                    "generator_wall_seconds": generator_seconds,
                    "generator_calls": generator.last_calls,
                    "generator_fallback": generator.last_fallback,
                    "gpu_seconds": target.last_gpu_seconds, "variant": args.variant}) + "\n")
                output.flush()
                os.fsync(output.fileno())


if __name__ == "__main__": main()
