#!/usr/bin/env python3
"""Train PPO on M_dev with evaluator-only case-specific scalar reward."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from gpu_runtime import add_gpu_arguments, configure_gpu
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from target_runner import VisionTarget
from adaptive_prompts import AdaptivePromptGenerator, TextAttacker
from evaluate_policy_phase2 import Selector
from pipeline_contract import validate_adapter, validate_target_protocol
from training_callbacks import StrategyTracker, make_training_callback
from public_feedback import reward as public_reward, refusal
from case_reward import assess as case_reward, incremental_value, load_dev_keys
from revision_contract import validate_target as validate_revision_target

MODES = ("text", "image", "cross_modal")


def ppo_batching(timesteps: int, requested_steps: int, requested_batch: int) -> tuple[int, int]:
    if timesteps < 2:
        raise ValueError("PPO needs at least two timesteps")
    n_steps = min(requested_steps, timesteps)
    divisors = [value for value in range(2, min(requested_batch, n_steps) + 1) if n_steps % value == 0]
    return n_steps, max(divisors, default=n_steps)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class CanaryEnv(gym.Env):
    """Multi-query M_dev episode with only public, filtered feedback."""
    def __init__(self, records, prompt_bank, target, generator, log_path: Path, policy_seed: int,
                 target_seed: int, partition_seed: int, max_tokens: int, variant: str,
                 episode_steps: int, dev_keys: dict[str, str] | None = None):
        self.records, self.bank, self.target, self.generator = records, prompt_bank, target, generator
        self.log_path, self.rng, self.max_tokens = log_path, random.Random(policy_seed), max_tokens
        self.log_handle = log_path.open("a", encoding="utf-8", buffering=1024 * 1024)
        self.log_rows = 0
        self.policy_seed, self.target_seed, self.partition_seed = policy_seed, target_seed, partition_seed
        self.variant, self.episode_steps, self.selector, self.step_index = variant, episode_steps, None, 0
        self.dev_keys = dev_keys
        self.action_space = spaces.MultiDiscrete([len(MODES), len(prompt_bank["text"])])
        # Generalizable public state: budget progress, last public response
        # signal, and three UCB statistics.  No canary ID or secret is observed.
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.current = None
        self.episode = 0
        self.record_order: list[int] = []
        self.record_cursor = 0

    def _next_record(self):
        if self.record_cursor >= len(self.record_order):
            self.record_order = list(range(len(self.records)))
            self.rng.shuffle(self.record_order)
            self.record_cursor = 0
        record = self.records[self.record_order[self.record_cursor]]
        self.record_cursor += 1
        return record

    def _observation(self):
        return self.selector.observation(self.step_index, self.episode_steps)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            self.record_order = []
            self.record_cursor = 0
        self.current = self._next_record()
        self.selector = Selector(self.variant, self.rng.randrange(2**31))
        self.step_index = 0
        self.used_prompts = set()
        self.history = []
        self.best_matching_digits = 0
        return self._observation(), {}

    def step(self, action):
        preferred = MODES[int(action[0])]
        mode = self.selector.choose(preferred)
        template_index = self.step_index % len(self.bank[mode]) if self.variant == "no_strategy_agent" else int(action[1])
        prompt, generator_seconds = self.generator.generate(
            self.current, mode, template_index, self.history, self.used_prompts)
        secret = self.dev_keys[self.current["canary_id"]] if self.dev_keys else None
        if secret and case_reward(prompt, secret).exact:
            raise ValueError("private answer appeared in a generated target prompt")
        image = Path(self.current["_root"]) / self.current["image"] if mode != "text" else None
        started = time.perf_counter()
        response = self.target.generate(prompt, image, self.max_tokens)
        elapsed = time.perf_counter() - started
        scored = case_reward(response, secret) if secret else None
        reward = (incremental_value(scored, self.best_matching_digits)
                  if scored else public_reward(response))
        if scored:
            self.best_matching_digits = max(self.best_matching_digits, scored.matching_digits)
        exact = scored.exact if scored else reward >= 8.0
        self.selector.update(mode, response)
        self.history.append({"prompt": prompt, "strategy_index": template_index, "response": response,
                             "feedback": self.selector.last_proxy,
                             "refusal": refusal(response)})
        self.step_index += 1
        self.log_handle.write(json.dumps({"canary_id": self.current["canary_id"], "method": "MM-LEAK-train",
                "target_seed": self.target_seed, "partition_seed": self.partition_seed,
                "policy_seed": self.policy_seed, "target_state": "lora_with_canary", "modality": mode,
                "family": self.target.family, "model": self.target.model_id,
                "query_index": self.step_index, "strategy_index": template_index,
                "strategy": self.bank[mode][template_index],
                "prompt": prompt, "response": response, "wall_seconds": elapsed,
                "generator_wall_seconds": generator_seconds,
                "generator_calls": self.generator.last_calls,
                "generator_fallback": self.generator.last_fallback,
                "gpu_seconds": self.target.last_gpu_seconds,
                "reward": reward, "exact_recovery": exact}) + "\n")
        self.log_rows += 1
        self.log_handle.flush()
        terminated = exact or self.step_index >= self.episode_steps
        if terminated: self.episode += 1
        info = {"public_format_hit": public_reward(response) >= 8.0,
                "exact_recovery": exact, "mode": mode, "modality": mode,
                "strategy_index": template_index, "strategy": self.bank[mode][template_index],
                "prompt": prompt, "response": response, "public_score": self.selector.last_proxy,
                "refusal": refusal(response), "success": exact}
        return self._observation(), reward, terminated, False, info

    def close(self):
        if not self.log_handle.closed:
            self.log_handle.flush()
            self.log_handle.close()
        super().close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--dev-keys", type=Path,
                        help="evaluator-private C_dev keys for case-specific reward")
    parser.add_argument("--reward-mode", choices=["keyed_v2", "public_v1"], default="keyed_v2")
    parser.add_argument("--adapter", required=True, type=Path); parser.add_argument("--target-plan", required=True, type=Path)
    parser.add_argument("--target-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-seed", required=True, type=int)
    parser.add_argument("--partition-seed", required=True, type=int)
    parser.add_argument("--policy-seed", required=True, type=int)
    parser.add_argument("--timesteps", type=int, default=10000)
    add_gpu_arguments(parser, required=False)
    parser.add_argument("--prompt-bank", required=True, type=Path); parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--attacker-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--attacker-revision")
    parser.add_argument("--attacker-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--attacker-max-new-tokens", type=int, default=160)
    parser.add_argument("--family", choices=["qwen2vl", "llava15"], default="qwen2vl"); parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--variant", choices=["full", "text_only", "image_only", "cross_modal_fixed", "random_modality", "no_ucb", "no_strategy_agent"], default="full")
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5); parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05); parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95); parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=2048); parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--strategy-update-episodes", type=int, default=5)
    args = parser.parse_args()
    if args.gpu_id is None: parser.error("online target calls require --gpu-id")
    if args.timesteps < 2: parser.error("--timesteps must be at least 2")
    if args.variant == "no_strategy_agent":
        parser.error("no_strategy_agent has no PPO to train; run its deterministic evaluation directly")
    if args.episode_steps != 100:
        parser.error("final and smoke policy runs must use the frozen 100-query episode horizon")
    if min(args.learning_rate, args.clip_range, args.gamma, args.gae_lambda) <= 0 or args.ent_coef < 0:
        parser.error("invalid PPO scalar hyperparameters")
    if args.gamma > 1 or args.gae_lambda > 1 or args.clip_range > 1:
        parser.error("gamma, gae-lambda and clip-range must not exceed 1")
    if min(args.n_epochs, args.n_steps, args.batch_size) < 1: parser.error("invalid PPO update hyperparameters")
    if args.strategy_update_episodes < 1: parser.error("--strategy-update-episodes must be positive")
    if args.reward_mode == "keyed_v2" and args.dev_keys is None:
        parser.error("keyed_v2 training requires --dev-keys")
    if args.reward_mode == "public_v1" and args.dev_keys is not None:
        parser.error("public_v1 training must not load --dev-keys")
    records = read_jsonl(args.queries)
    if not records or any(record.get("split") != "dev" for record in records):
        raise ValueError("Phase-II policy training accepts C_dev public queries only")
    if any(int(record.get("partition_seed", record.get("seed", -1))) != args.partition_seed for record in records):
        raise ValueError("Phase-II partition seed does not match C_dev")
    dev_keys = (load_dev_keys(args.dev_keys, records, args.target_seed, args.partition_seed)
                if args.dev_keys else None)
    for record in records: record["_root"] = str(args.queries.parent)
    bank = json.loads(args.prompt_bank.read_text(encoding="utf-8"))
    target_plan = json.loads(args.target_plan.read_text(encoding="utf-8"))
    if target_plan.get("protocol") == "revision_v1":
        validate_revision_target(args.target_plan, args.adapter, args.target_protocol,
                                 split="dev", target_seed=args.target_seed,
                                 partition_seed=args.partition_seed, model=args.model)
    else:
        planned_adapter = Path(target_plan["output"]) / "implanted" / "adapter"
    if target_plan.get("protocol") != "revision_v1" and (target_plan.get("status") != "validated" or target_plan.get("split") != "dev"):
        raise ValueError("Phase-II training requires a validated M_dev finetune plan")
    if target_plan.get("protocol") != "revision_v1" and "implanted" not in target_plan.get("arms", []): raise ValueError("M_dev plan does not include the implanted arm")
    if int(target_plan.get("target_seed", target_plan.get("seed", -1))) != args.target_seed or target_plan.get("family") != args.family or target_plan.get("model") != args.model:
        raise ValueError("M_dev plan seed/family/model differs from policy training")
    if target_plan.get("queries_sha256") is not None and target_plan["queries_sha256"] != hashlib.sha256(args.queries.read_bytes()).hexdigest():
        raise ValueError("public dev queries changed after target-plan freeze")
    if target_plan.get("protocol") != "revision_v1" and planned_adapter.resolve() != args.adapter.resolve():
        raise ValueError("--adapter is not the implanted adapter declared by the M_dev plan")
    if not (args.adapter / "adapter_config.json").exists(): raise FileNotFoundError("M_dev adapter is incomplete")
    if target_plan.get("protocol") != "revision_v1":
        validate_adapter(args.target_plan, args.adapter, "implanted", split="dev", family=args.family,
                         model=args.model, seed=args.target_seed)
        validate_target_protocol(args.target_protocol, args.target_plan, "dev")
    if set(bank) != set(MODES) or len({len(bank[m]) for m in MODES}) != 1: raise ValueError("prompt bank must have equal text/image/cross_modal lists")
    gpu_info = configure_gpu(args.gpu_id, args.disable_cudnn)
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    random.seed(args.policy_seed); np.random.seed(args.policy_seed); torch.manual_seed(args.policy_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.policy_seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to train into non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "training_query_log.jsonl"
    target = VisionTarget(args.family, args.model, args.adapter, "cuda")
    attacker = TextAttacker(args.attacker_model, args.attacker_device,
                            args.attacker_max_new_tokens, args.attacker_revision)
    generator = AdaptivePromptGenerator(attacker, bank)
    env = Monitor(CanaryEnv(records, bank, target, generator, log_path, args.policy_seed,
                            args.target_seed, args.partition_seed, args.max_new_tokens,
                            args.variant, args.episode_steps, dev_keys))
    n_steps, batch_size = ppo_batching(args.timesteps, args.n_steps, args.batch_size)
    policy = PPO("MlpPolicy", env, seed=args.policy_seed, n_steps=n_steps, batch_size=batch_size, device="cpu",
                 learning_rate=args.learning_rate, clip_range=args.clip_range, ent_coef=args.ent_coef,
                 gamma=args.gamma, gae_lambda=args.gae_lambda, n_epochs=args.n_epochs, verbose=1)
    tracker = StrategyTracker(args.output, bank, "phase2")
    callback = make_training_callback(tracker, attacker, args.timesteps,
                                      args.strategy_update_episodes)
    try:
        policy.learn(total_timesteps=args.timesteps, callback=callback)
        policy.save(args.output / "policy")
        actual_timesteps = int(policy.num_timesteps)
    finally:
        tracker.close()
        env.close()
    manifest = {**vars(args), "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        "gpu_runtime": gpu_info,
        "policy_sha256": hashlib.sha256((args.output / "policy.zip").read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).with_name("protected_interface.py").read_bytes()).hexdigest(),
        "public_feedback_sha256": hashlib.sha256(Path(__file__).with_name("public_feedback.py").read_bytes()).hexdigest(),
        "case_reward_sha256": hashlib.sha256(Path(__file__).with_name("case_reward.py").read_bytes()).hexdigest(),
        "dev_keys_sha256": hashlib.sha256(args.dev_keys.read_bytes()).hexdigest() if args.dev_keys else None,
        "target_plan_sha256": hashlib.sha256(args.target_plan.read_bytes()).hexdigest(),
        "target_protocol_sha256": hashlib.sha256(args.target_protocol.read_bytes()).hexdigest(),
        "prompt_bank_sha256": hashlib.sha256(args.prompt_bank.read_bytes()).hexdigest(),
        "learned_strategy_bank_sha256": tracker.bank_sha256(),
        "strategy_generations": tracker.generation,
        "n_steps": n_steps, "batch_size_effective": batch_size,
        "actual_timesteps": actual_timesteps}
    manifest["prompt_method"] = "online_adaptive_generator_evolving_v3"
    manifest["attacker_resolved_revision"] = attacker.resolved_revision
    (args.output / "frozen_policy_manifest.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")


if __name__ == "__main__": main()
