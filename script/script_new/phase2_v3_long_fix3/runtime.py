"""Query execution and a secret-free, versioned controller environment."""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import ROOT, digest, no_placeholders, normalize, sha256
from .feedback import feedback
from .strategies import STRATEGIES, StrategySpec, render

OBS_VERSION = "public-hash-64x3-v1"
OBS_DIM = 64 * 3 + 4 + 12 + 12 + 8


def suffix_decode(tokenizer, generated_ids, input_length: int) -> tuple[str, int, str]:
    """Decode only new tokens; works with torch tensors and list-backed test doubles."""
    output = generated_ids[0]
    tail = output[input_length:]
    text = tokenizer.decode(tail, skip_special_tokens=True).strip()
    eos = getattr(tokenizer, "eos_token_id", None)
    reason = "eos" if eos is not None and len(tail) and int(tail[-1]) == int(eos) else "length_or_other"
    return text, len(tail), reason


def hash_embedding(text: str, dim: int = 64) -> list[float]:
    """Fixed public text embedding, stable across processes and GPU versions."""
    values = [0.0] * dim
    words = normalize(text).split()
    for word in words[:512]:
        h = hashlib.sha256(word.encode()).digest()
        index = int.from_bytes(h[:4], "big") % dim
        values[index] += 1.0 if h[4] & 1 else -1.0
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values] if norm else values


def observation(cue: str, remaining: int, budget: int, prior_action: tuple[int, int, int] | None,
                last_prompt: str, last_response: str, flags: dict, used: set[int]) -> list[float]:
    features = (hash_embedding(cue) + hash_embedding(last_prompt) + hash_embedding(last_response)
                + [remaining / budget, float(flags.get("refusal_detected", False)),
                   float(flags.get("partial_format", False)), float(flags.get("valid_candidate_count", 0) > 0)])
    features += [float(prior_action is not None and prior_action[0] == i) for i in range(12)]
    features += [float(i in used) for i in range(12)]  # observation feature, not distribution mask
    features += ([float(prior_action is not None and prior_action[1] == i) for i in range(5)]
                 + [float(prior_action is not None and prior_action[2] == i) for i in range(3)])
    assert len(features) == OBS_DIM
    return features


class HistoricalTarget:
    """Exact protected wrapper from the historical runner, imported only on GPU."""
    def __init__(self, adapter: Path):
        from target_attacker_phase2 import Qwen2VLTarget_LoRaCorrected
        import torch
        prior = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            self.model = Qwen2VLTarget_LoRaCorrected(adapter)
        finally:
            torch.backends.cudnn.enabled = prior
        self.torch = torch
        off_gpu = [name for name, parameter in self.model.model.named_parameters()
                   if parameter.device.type != "cuda"]
        if off_gpu:
            raise RuntimeError(f"Target parameters off CUDA: {off_gpu[:5]}")
        self.metadata = {"adapter": str(adapter),
                         "model_revision": getattr(self.model.model.config, "_commit_hash", None),
                         "accepts_mm_token_type_ids": self.model.accepts_mm_token_type_ids}

    def sanity(self) -> dict:
        started = time.monotonic()
        answer = self.query("Compute 2 + 2. Reply with just the numeral.", "text")
        seconds = time.monotonic() - started
        if "4" not in answer or len(answer) > 80 or seconds > 30:
            raise RuntimeError(f"Target sanity probe failed: {answer!r}, {seconds:.1f}s")
        return {"answer": answer, "seconds": seconds, "target_calls": 1}

    def query(self, data, modality: str, suffix: str = "") -> str:
        prior = self.torch.backends.cudnn.enabled
        self.torch.backends.cudnn.enabled = False
        try:
            if modality == "text":
                return self.model.process_input(data, input_type="text")
            if modality in ("image", "image_text"):
                return self.model.process_input(data, input_type="image", prompt_suffix=suffix)
            raise ValueError(f"Bad modality {modality}")
        finally:
            self.torch.backends.cudnn.enabled = prior


class FakeTarget:
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["I cannot disclose private access keys."]
        self.calls = []

    def sanity(self) -> dict:
        return {"answer": "4", "seconds": 0.0, "target_calls": 0}

    def query(self, data, modality: str, suffix: str = "") -> str:
        self.calls.append({"modality": modality, "data": data, "suffix": suffix})
        return self.responses[(len(self.calls) - 1) % len(self.responses)]


class QueryEngine:
    def __init__(self, target, run_id: str, log_path: Path | None, max_retries: int = 2,
                 max_target_calls: int | None = None):
        self.target, self.run_id, self.log_path = target, run_id, log_path
        self.max_retries = max_retries
        self.max_target_calls = max_target_calls
        self.logged = {}
        self.attempts = {}
        if log_path and log_path.exists():
            for line in log_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    qid = row["query_id"]
                    if qid in self.logged:
                        raise ValueError(f"Duplicate query ID in log: {qid}")
                    self.logged[qid] = row
        if log_path:
            attempt_path = log_path.parent / (log_path.stem + "_target_attempts.jsonl")
            if attempt_path.exists():
                for line in attempt_path.read_text().splitlines():
                    if line.strip():
                        item = json.loads(line)
                        qid = item["query_id"]
                        if qid in self.attempts:
                            raise ValueError(f"Duplicate target attempt ID: {qid}")
                        self.attempts[qid] = item

    def _record_error(self, query_id: str, stage: str, exc: Exception) -> None:
        if not self.log_path:
            return
        error_path = self.log_path.parent / (self.log_path.stem + "_errors.jsonl")
        with error_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"query_id": query_id, "stage": stage,
                                     "error_type": type(exc).__name__, "error": str(exc),
                                     "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")

    def query(self, case: dict, episode: int, step: int, action: tuple[int, int, int],
              history: list[dict], seen_prompts: set[str], seen_candidates: set[str],
              seed: int) -> dict:
        strategy, pressure, framing = map(int, action)
        if strategy not in range(len(STRATEGIES)) or pressure not in range(5) or framing not in range(3):
            raise ValueError("Invalid action")
        qid = f"{self.run_id}:{case['case_id']}:{episode}:{step}"
        if qid in self.logged:
            row = self.logged[qid]
            if row["action"] != list(action):
                raise RuntimeError(f"Replay action differs for {qid}; no target requery")
            seen_prompts.add(normalize(row["prompt"]))
            seen_candidates.update(row["public_feedback"]["candidate_values"])
            return row
        if qid in self.attempts:
            raise RuntimeError(f"Unresolved target-call intent for {qid}; refusing a duplicate query")
        if self.max_target_calls is not None and len(self.attempts) >= self.max_target_calls:
            raise RuntimeError("Target-call budget exhausted before query")
        spec = STRATEGIES[strategy]
        render_start = time.monotonic()
        rendered = None
        retries = 0
        for attempt in range(self.max_retries + 1):
            try:
                # Reserve a disjoint revision range for each step. PPO may repeat
                # the same action throughout an episode; a fixed 0..2 revision
                # range would otherwise exhaust after three uses.
                revision = (step - 1) * (self.max_retries + 1) + attempt
                rendered = render(spec, case["prefix"], pressure, framing,
                                  history[-3:], revision)
            except Exception as exc:
                self._record_error(qid, "render", exc)
                raise
            if not rendered.text or not no_placeholders(rendered.text):
                exc = ValueError("Malformed rendered prompt; no fallback")
                self._record_error(qid, "validate_prompt", exc)
                raise exc
            key = normalize(rendered.text + (rendered.image_text or ""))
            if key not in seen_prompts:
                break
            retries += 1
        else:
            exc = RuntimeError("Duplicate prompt retry limit reached; no fallback")
            self._record_error(qid, "deduplicate", exc)
            raise exc
        assert rendered is not None
        seen_prompts.add(key)
        render_seconds = time.monotonic() - render_start
        image_path = None
        if rendered.image is not None and self.log_path:
            image_dir = self.log_path.parent / (self.log_path.stem + "_images")
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / (digest(qid)[:20] + ".png")
            if image_path.exists():
                if sha256(image_path) != rendered.image_sha256:
                    raise ValueError("Existing rendered image hash mismatch")
            else:
                try:
                    rendered.image.save(image_path, format="PNG")
                except Exception as exc:
                    self._record_error(qid, "save_image", exc)
                    raise
                if sha256(image_path) != rendered.image_sha256:
                    raise RuntimeError("Rendered image changed during save")
        start = time.monotonic()
        if self.log_path:
            intent = {"query_id": qid, "case_id": case["case_id"], "episode": episode,
                      "step": step, "at_utc": datetime.now(timezone.utc).isoformat()}
            attempt_path = self.log_path.parent / (self.log_path.stem + "_target_attempts.jsonl")
            with attempt_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(intent) + "\n")
                stream.flush()
                import os
                os.fsync(stream.fileno())
            self.attempts[qid] = intent
        try:
            if rendered.modality == "text":
                response = self.target.query(rendered.text, "text")
            else:
                response = self.target.query(rendered.image, rendered.modality, rendered.text)
        except Exception as exc:
            self._record_error(qid, "target_inference", exc)
            raise
        target_seconds = time.monotonic() - start
        if not isinstance(response, str):
            exc = TypeError("Target returned non-text response")
            self._record_error(qid, "validate_target_response", exc)
            raise exc
        result = feedback(response, seen_candidates, duplicate_prompt=bool(retries))
        row = {
            "run_id": self.run_id, "query_id": qid, "case_id": case["case_id"],
            "episode_id": episode, "step": step, "strategy_id": spec.id,
            "history_entries_seen": min(len(history), 3),
            "requested_modality": spec.modality,
            "effective_modality": "text" if rendered.image is None else "image",
            "action": list(action), "prompt": rendered.text,
            "image_text": rendered.image_text, "image_sha256": rendered.image_sha256,
            "font_sha256": rendered.font_sha256, "renderer": rendered.renderer,
            "response_after_filter": response, "public_feedback": result,
            "reward": result["reward"], "policy_seed": seed,
            "generation_seed": int(digest(f"{seed}:{qid}")[:16], 16),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "image_path": str(image_path.relative_to(self.log_path.parent)) if image_path else None,
            "generation": {"method": "deterministic_document_v1", "token_count": len(rendered.text.split()),
                           "stop_reason": "template_complete", "retry_count": retries,
                           "attacker_calls": retries + 1},
            "tokens": {"attacker_prompt_whitespace_estimate": len(rendered.text.split()),
                       "attacker_model_generated": 0,
                       "target_response_whitespace_estimate": len(response.split()),
                       "target_generated_exact": None,
                       "target_stop_reason": None},
            "cost": {"target_calls": 1, "attacker_calls": retries + 1, "image_model_calls": 0},
            "timing": {"render_seconds": render_seconds, "target_seconds": target_seconds},
        }
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            self.logged[qid] = row
        return row


class ControllerEnv:
    """Gym wrapper is created lazily so imports and dry runs never need SB3."""
    def __init__(self, cases: list[dict], engine: QueryEngine, budget: int, seed: int,
                 selection: str = "random"):
        if budget < 1 or not cases:
            raise ValueError("Empty cases or invalid budget")
        if any(set(c) != {"case_id", "prefix"} for c in cases):
            raise ValueError("Controller accepts only case_id and public prefix")
        self.cases, self.engine, self.budget, self.seed = cases, engine, budget, seed
        self.selection = selection
        self.rng = random.Random(seed)
        self.episode = 0
        self.case = None
        self.step_index = 0
        self.history = []
        self.seen_prompts = set()
        self.seen_candidates = set()
        self.used = set()
        self.last_action = None
        self.last_feedback = {}

    def reset(self, *, case: dict | None = None):
        self.episode += 1
        self.case = case or self.rng.choice(self.cases)
        self.step_index = 0
        self.history = []
        self.seen_prompts = set()
        self.seen_candidates = set()
        self.used = set()
        self.last_action = None
        self.last_feedback = {}
        return self._obs(), {"case_id": self.case["case_id"]}

    def _obs(self):
        values = observation(self.case["prefix"], self.budget - self.step_index,
                             self.budget, self.last_action,
                             ((self.history[-1].get("image_text") or "") + " " +
                              self.history[-1]["prompt"]) if self.history else "",
                             self.history[-1]["response_after_filter"] if self.history else "",
                             self.last_feedback, self.used)
        try:
            import numpy as np
        except ImportError:
            return values
        return np.asarray(values, dtype=np.float32)

    def step(self, action):
        if self.case is None or self.step_index >= self.budget:
            raise RuntimeError("Reset before stepping")
        action = tuple(int(x) for x in action)
        self.step_index += 1
        row = self.engine.query(self.case, self.episode, self.step_index, action, self.history,
                                self.seen_prompts, self.seen_candidates, self.seed)
        self.history.append({"prompt": row["prompt"],
                             "image_text": row["image_text"],
                             "image_sha256": row["image_sha256"],
                             "response_after_filter": row["response_after_filter"]})
        self.used.add(action[0])
        self.last_action = action
        self.last_feedback = row["public_feedback"]
        return self._obs(), float(row["reward"]), False, self.step_index >= self.budget, row


def gym_env(base: ControllerEnv):
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    class Adapter(gym.Env):
        action_space = spaces.MultiDiscrete([len(STRATEGIES), 5, 3])
        observation_space = spaces.Box(-2.0, 2.0, shape=(OBS_DIM,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return base.reset(case=(options or {}).get("case"))

        def step(self, action):
            return base.step(action)

    return Adapter()
