#!/usr/bin/env python3
"""Phase I: adaptive PII-like elicitation on public, canary-free form tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from gpu_runtime import add_gpu_arguments, configure_gpu
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from target_runner import VisionTarget
from adaptive_prompts import AdaptivePromptGenerator, TextAttacker
from phase1_feedback import assess
from prepare_phase1_data import validate_phase1_records
from training_callbacks import StrategyTracker, make_training_callback

MODES = ("text", "image", "cross_modal")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ppo_batching(timesteps: int, requested_steps: int, requested_batch: int) -> tuple[int, int]:
    if timesteps < 2:
        raise ValueError("PPO needs at least two timesteps")
    n_steps = min(requested_steps, timesteps)
    divisors = [value for value in range(2, min(requested_batch, n_steps) + 1) if n_steps % value == 0]
    return n_steps, max(divisors, default=n_steps)


class Phase1Env(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, records, bank, target, generator, output: Path, seed: int, target_state: str, max_tokens: int, episode_steps: int = 5):
        self.records, self.bank, self.target, self.generator = records, bank, target, generator
        self.output, self.rng, self.seed_value = output, random.Random(seed), seed
        self.log_handle = output.open("a", encoding="utf-8", buffering=1024 * 1024)
        self.log_rows = 0
        self.target_state, self.max_tokens = target_state, max_tokens
        self.action_space = spaces.MultiDiscrete([len(MODES), len(bank["text"])])
        self.observation_space = spaces.Box(0, 1, shape=(4,), dtype=np.float32)
        self.episode_steps = episode_steps
        self.current = None
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

    def _obs(self):
        return np.array([self.step_index / self.episode_steps, self.last_valid,
                         self.last_refused, self.last_feedback], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            self.record_order = []
            self.record_cursor = 0
        self.current = self._next_record()
        self.step_index, self.last_valid, self.last_refused, self.last_feedback = 0, 0., 0., 0.
        self.history, self.used_prompts = [], set()
        return self._obs(), {}

    def step(self, action):
        mode, template_index = MODES[int(action[0])], int(action[1])
        prompt, generator_seconds = self.generator.generate(
            self.current, mode, template_index, self.history, self.used_prompts)
        image = Path(self.current["_root"]) / self.current["image"] if mode != "text" else None
        started = time.perf_counter(); response = self.target.generate(prompt, image, self.max_tokens)
        elapsed = time.perf_counter() - started
        metrics = assess(response)
        reward = metrics["reward"]
        self.step_index += 1
        self.last_valid, self.last_refused = float(metrics["pii_like"]), float(metrics["refusal"])
        self.last_feedback = metrics["feedback"]
        self.history.append({"prompt": prompt, "strategy_index": template_index, "response": response,
                             "feedback": self.last_feedback, "refusal": metrics["refusal"]})
        self.log_handle.write(json.dumps({"phase": "PII-like content elicitation", "task_id": self.current["task_id"],
                "method": "MM-LEAK-phase1", "policy_seed": self.seed_value, "target_state": self.target_state,
                "family": self.target.family, "model": self.target.model_id,
                "modality": mode, "query_index": self.step_index, "strategy_index": template_index,
                "strategy": self.bank[mode][template_index],
                "prompt": prompt, "response": response, "generator_wall_seconds": generator_seconds,
                "generator_calls": self.generator.last_calls,
                "generator_fallback": self.generator.last_fallback,
                "wall_seconds": elapsed, "gpu_seconds": self.target.last_gpu_seconds,
                "pii_like": metrics["pii_like"], "pii_types": metrics["pii_types"],
                "refusal": metrics["refusal"], "reward": reward}) + "\n")
        self.log_rows += 1
        self.log_handle.flush()
        info = {**metrics, "modality": mode, "strategy_index": template_index,
                "strategy": self.bank[mode][template_index], "prompt": prompt,
                "response": response, "public_score": metrics["feedback"],
                "success": metrics["pii_like"]}
        return self._obs(), reward, self.step_index >= self.episode_steps, False, info

    def close(self):
        if not self.log_handle.closed:
            self.log_handle.flush()
            self.log_handle.close()
        super().close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--policy-seed", required=True, type=int)
    parser.add_argument("--timesteps", type=int, default=5000); parser.add_argument("--prompt-bank", required=True, type=Path)
    parser.add_argument("--attacker-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--attacker-revision")
    parser.add_argument("--attacker-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--attacker-max-new-tokens", type=int, default=160)
    parser.add_argument("--episode-steps", type=int, default=5)
    add_gpu_arguments(parser, required=False)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--family", choices=["qwen2vl", "llava15"], default="qwen2vl"); parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5); parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05); parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95); parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=1024); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--strategy-update-episodes", type=int, default=5)
    args = parser.parse_args()
    if args.gpu_id is None: parser.error("online target calls require --gpu-id")
    if args.episode_steps < 2: parser.error("--episode-steps must be at least 2")
    if args.timesteps < 2: parser.error("--timesteps must be at least 2")
    if min(args.learning_rate, args.clip_range, args.gamma, args.gae_lambda) <= 0 or args.ent_coef < 0:
        parser.error("invalid PPO scalar hyperparameters")
    if args.gamma > 1 or args.gae_lambda > 1 or args.clip_range > 1:
        parser.error("gamma, gae-lambda and clip-range must not exceed 1")
    if min(args.n_epochs, args.n_steps, args.batch_size) < 1: parser.error("invalid PPO update hyperparameters")
    if args.strategy_update_episodes < 1: parser.error("--strategy-update-episodes must be positive")
    bank = json.loads(args.prompt_bank.read_text(encoding="utf-8"))
    if set(bank) != set(MODES) or len({len(bank[mode]) for mode in MODES}) != 1 or any(not bank[mode] for mode in MODES):
        raise ValueError("Phase-I strategy bank must have equal nonempty modality lists")
    if "canary" in json.dumps(bank).lower():
        raise ValueError("Phase-I strategies must not mention canaries")
    records = read_jsonl(args.queries)
    validate_phase1_records(records, "dev", args.policy_seed)
    for record in records: record["_root"] = str(args.queries.parent)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to train into non-empty output directory: {args.output}")
    gpu_info = configure_gpu(args.gpu_id, args.disable_cudnn)
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "phase1_train_log.jsonl"
    random.seed(args.policy_seed); np.random.seed(args.policy_seed)
    import torch
    torch.manual_seed(args.policy_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.policy_seed)
    target = VisionTarget(args.family, args.model, None, "cuda", None)
    attacker = TextAttacker(args.attacker_model, args.attacker_device,
                            args.attacker_max_new_tokens, args.attacker_revision)
    generator = AdaptivePromptGenerator(attacker, bank, goal="pii_elicitation")
    env = Monitor(Phase1Env(records, bank, target, generator, log_path, args.policy_seed,
                            "base", args.max_new_tokens, args.episode_steps))
    n_steps, batch_size = ppo_batching(args.timesteps, args.n_steps, args.batch_size)
    model = PPO("MlpPolicy", env, seed=args.policy_seed, n_steps=n_steps, batch_size=batch_size, device="cpu",
                learning_rate=args.learning_rate, clip_range=args.clip_range, ent_coef=args.ent_coef,
                gamma=args.gamma, gae_lambda=args.gae_lambda, n_epochs=args.n_epochs, verbose=1)
    tracker = StrategyTracker(args.output, bank, "phase1")
    callback = make_training_callback(tracker, attacker, args.timesteps,
                                      args.strategy_update_episodes)
    try:
        model.learn(args.timesteps, callback=callback)
        model.save(args.output / "phase1_policy")
        actual_timesteps = int(model.num_timesteps)
    finally:
        tracker.close()
        env.close()
    manifest = {**vars(args), "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
                "gpu_runtime": gpu_info,
                "policy_sha256": hashlib.sha256((args.output / "phase1_policy.zip").read_bytes()).hexdigest(),
                "wrapper_sha256": hashlib.sha256(Path(__file__).with_name("protected_interface.py").read_bytes()).hexdigest(),
                "prompt_bank_sha256": hashlib.sha256(args.prompt_bank.read_bytes()).hexdigest(),
                "learned_strategy_bank_sha256": tracker.bank_sha256(),
                "strategy_generations": tracker.generation,
                "n_steps": n_steps, "batch_size_effective": batch_size,
                "actual_timesteps": actual_timesteps}
    manifest["prompt_method"] = "phase1_pii_online_evolving_v2"
    manifest["target_state"] = "base"
    manifest["attacker_resolved_revision"] = attacker.resolved_revision
    (args.output / "phase1_manifest.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")


if __name__ == "__main__": main()
