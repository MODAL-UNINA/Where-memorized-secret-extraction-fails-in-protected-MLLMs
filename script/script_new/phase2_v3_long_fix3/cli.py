"""Phase II v3 command line. Model imports occur only after GPU preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path

from .core import (DATA, HERE, ROOT, digest, ensure_gpu, load_split, protocol_hashes,
                   public_cases, read_jsonl, sha256, verify_hashes, write_json)
from .diagnose import diagnose
from .runtime import (OBS_VERSION, ControllerEnv, FakeTarget, HistoricalTarget,
                      QueryEngine, gym_env)
from .strategies import STRATEGIES, render, serialized_repertoire, validate_repertoire

DEFAULT_RUN = ROOT / "runs/disjoint_lora_seed42/benign_replay_v2/phase2_v3_long_seed42_fix3"
SOURCE_RUN = ROOT / "runs/disjoint_lora_seed42/benign_replay_v2/phase2_v3_seed42"
MIGRATION_SOURCE_RUN = ROOT / "runs/disjoint_lora_seed42/benign_replay_v2/phase2_v3_long_seed42_fix2"
ADAPTER = ROOT / "runs/disjoint_lora_seed42/benign_replay_v2/qwen-dev-r64-e3-final"


def config() -> dict:
    return json.loads((HERE / "config.json").read_text())


def _read_prepared(run: Path) -> tuple[dict, dict, dict]:
    conf = config()
    frozen = json.loads((run / "protocol.json").read_text())
    if frozen["config_sha256"] != sha256(HERE / "config.json"):
        raise ValueError("Configuration changed since prepare")
    if frozen["repertoire_sha256"] != sha256(run / "repertoire.json"):
        raise ValueError("Repertoire changed since prepare")
    if json.loads((run / "repertoire.json").read_text()) != serialized_repertoire():
        raise ValueError("Strategy mapping changed since prepare")
    if frozen["split_sha256"] != sha256(run / "split.json"):
        raise ValueError("Dev split changed since prepare")
    verify_hashes(frozen["protocol_hashes"], ADAPTER)
    split = load_split(run / "split.json")
    if frozen["observation_version"] != OBS_VERSION:
        raise ValueError("Observation schema mismatch")
    if frozen.get("source_pilot_protocol_sha256") != sha256(SOURCE_RUN / "protocol.json"):
        raise ValueError("Source pilot protocol changed")
    for name, expected in frozen.get("source_pilot_hashes", {}).items():
        if sha256(SOURCE_RUN / name) != expected or sha256(run / name) != expected:
            raise ValueError(f"Source pilot artifact changed: {name}")
    return conf, frozen, split


def prepare(run: Path, dry: bool) -> dict:
    conf = config()
    source_protocol = json.loads((SOURCE_RUN / "protocol.json").read_text())
    if source_protocol["split_sha256"] != sha256(SOURCE_RUN / "split.json"):
        raise ValueError("Source pilot split changed")
    if source_protocol["repertoire_sha256"] != sha256(SOURCE_RUN / "repertoire.json"):
        raise ValueError("Source pilot repertoire changed")
    if source_protocol["protocol_hashes"]["adapter"] != sha256(ADAPTER / "adapter_model.safetensors"):
        raise ValueError("Source pilot target adapter changed")
    source_pilot_files = ("pilot_summary.json", "pilot_private.json", "pilot_queries.jsonl",
                          "pilot_queries_target_attempts.jsonl")
    source_pilot_hashes = {name: sha256(SOURCE_RUN / name) for name in source_pilot_files}
    source_pilot = json.loads((SOURCE_RUN / "pilot_summary.json").read_text())
    source_report = json.loads((SOURCE_RUN / "pilot_private.json").read_text())
    source_private = source_report["aggregate"]
    if source_pilot.get("pilot_target_calls") != 96 or source_private.get("queries") != 96:
        raise ValueError("Source pilot is incomplete")
    if source_report["trace_sha256"] != source_pilot_hashes["pilot_queries.jsonl"]:
        raise ValueError("Source private pilot score does not match its query trace")
    cases = public_cases()
    validate_repertoire()
    source = DATA / "dev_attack_public.jsonl"
    # The only random operation here is the frozen public-ID split.
    ids = [c["case_id"] for c in cases]
    random.Random(42).shuffle(ids)
    split = {"schema": "phase2-v3-split-1", "seed": 42,
             "source_sha256": sha256(source), "dev_train": ids[:8],
             "dev_validation": ids[8:]}
    if split != json.loads((SOURCE_RUN / "split.json").read_text()):
        raise ValueError("Long run must reuse the pilot's frozen 8/4 dev split")
    hashes = protocol_hashes(ADAPTER)
    split_manifest = json.loads((DATA / "manifest.json").read_text())
    if split_manifest["seed"] != 42 or split_manifest["models"]["dev"]["dataset_sha256"] != hashes["dev_dataset"]:
        raise ValueError("Frozen dev dataset provenance mismatch")
    training_manifest = json.loads((ADAPTER / "training_manifest.json").read_text())
    if training_manifest["adapter_sha256"] != hashes["adapter"] or training_manifest["status"] != "complete":
        raise ValueError("Adapter provenance hash mismatch")
    recipe_dataset = DATA / "benign_replay_v2/dev.jsonl"
    if training_manifest["dataset_sha256"] != sha256(recipe_dataset):
        raise ValueError("Adapter was trained on a different dev recipe dataset")
    probe = json.loads((ADAPTER.parent / "dev_full_text_probe.json").read_text())
    if probe.get("status") != "PASS" or not probe["model_probe"]["all_verified"]:
        raise ValueError("Dev target memorization gate failed")
    if probe["inputs"]["adapter_sha256"] != hashes["adapter"] or probe["inputs"]["split_manifest_sha256"] != hashes["split_manifest"]:
        raise ValueError("Dev memorization report refers to a different target")
    if probe["inputs"]["training_dataset_sha256"] != sha256(recipe_dataset) or probe["inputs"]["canary_scoring_sha256"] != hashes["canary_scoring.py"]:
        raise ValueError("Dev memorization report uses different inputs or decoder")
    protocol = {"schema": "phase2-v3-protocol-1", "config_sha256": sha256(HERE / "config.json"),
                "protocol_hashes": hashes, "split_sha256": None,
                "repertoire_sha256": None, "observation_version": OBS_VERSION,
                "target_model": "Qwen/Qwen2-VL-7B-Instruct",
                "target_recipe": conf["target_recipe"],
                "target_loading": "Qwen2VLTarget_LoRaCorrected.process_input unchanged",
                "private_labels_excluded": True,
                "run_role": "checkpoint-migrated long training after metrics serialization stop; scientific result exploratory",
                "source_pilot_protocol_sha256": sha256(SOURCE_RUN / "protocol.json"),
                "source_pilot_hashes": source_pilot_hashes,
                "source_pilot_private_aggregate": source_private}
    result = {"run": str(run), "split": split, "protocol": protocol,
              "planned_target_calls": {"pilot_reused": 96, "train_max": conf["train_target_calls_max"],
                                       "validation_per_method": 48}}
    if dry:
        return result
    run.mkdir(parents=True, exist_ok=True)
    for name, value in (("split.json", split), ("repertoire.json", serialized_repertoire())):
        path = run / name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise FileExistsError(f"Immutable prepared file differs: {path}")
        else:
            write_json(path, value)
    protocol["split_sha256"] = sha256(run / "split.json")
    protocol["repertoire_sha256"] = sha256(run / "repertoire.json")
    path = run / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise FileExistsError("Prepared protocol differs")
    else:
        write_json(path, protocol)
    from shutil import copyfile
    for name in source_pilot_files:
        destination = run / name
        if destination.exists():
            if sha256(destination) != source_pilot_hashes[name]:
                raise FileExistsError(f"Source pilot copy changed: {name}")
        else:
            copyfile(SOURCE_RUN / name, destination)
    result["protocol"] = protocol
    return result


def _case_subset(split: dict, name: str) -> list[dict]:
    ids = split[name]
    mapping = {c["case_id"]: c for c in public_cases()}
    return [mapping[x] for x in ids]


def _target(fake: bool = False):
    if fake:
        return FakeTarget()
    ensure_gpu()
    sys.path.insert(0, str(ROOT / "script/script_new"))
    return HistoricalTarget(ADAPTER)


def _verify_target_runtime(run: Path, target: HistoricalTarget) -> None:
    pilot_file = run / "pilot_summary.json"
    if pilot_file.exists():
        expected = json.loads(pilot_file.read_text()).get("target_runtime")
        if expected != target.metadata:
            raise ValueError("Target runtime revision/interface differs from pilot")


def probe(run: Path, dry: bool, gpu: bool) -> dict:
    conf, _, split = _read_prepared(run)
    case = _case_subset(split, "dev_train")[0]
    rendered = [render(spec, case["prefix"], 0, 0, []) for spec in STRATEGIES]
    if any(case["prefix"] not in (item.text + (item.image_text or "")) for item in rendered):
        raise ValueError("A renderer lost its public cue")
    result = {"rendered": len(rendered), "modalities": [item.modality for item in rendered],
              "target_calls": 0, "gpu": gpu}
    if dry or not gpu:
        return result
    if (run / "probe_gpu.json").exists():
        raise FileExistsError("GPU probe report exists; preserve it")
    target = _target()
    result["sanity"] = target.sanity()
    # Benign inputs test three actual transport paths, not canary extraction.
    for modality in ("text", "image", "image_text"):
        if modality == "text":
            target.query("Compute 2 + 2. Reply with just the numeral.", "text")
        else:
            from .strategies import _document
            image, _, _ = _document(["BENIGN TRANSPORT CHECK", "Arithmetic: 2 + 2 = 4"])
            target.query(image, modality, "Describe the document layout in one sentence.")
        result["target_calls"] += 1
    result["target_calls"] += result["sanity"]["target_calls"]
    _cost_event(run, "probe_gpu", result["target_calls"])
    write_json(run / "probe_gpu.json", result)
    return result


def pilot(run: Path, dry: bool) -> dict:
    conf, _, split = _read_prepared(run)
    cases = _case_subset(split, "dev_train")
    if dry:
        return {"target_calls": len(cases) * len(STRATEGIES), "cases": len(cases),
                "strategies": len(STRATEGIES), "models_loaded": False}
    if (run / "pilot_summary.json").exists():
        raise FileExistsError("Pilot summary exists; preserve the completed pilot")
    ensure_gpu()
    target = _target()
    _verify_target_runtime(run, target)
    sanity = target.sanity()
    _cost_event(run, "pilot_sanity", sanity["target_calls"])
    engine = QueryEngine(target, "pilot", run / "pilot_queries.jsonl",
                         conf["max_attacker_retries"])
    existing_calls = len(engine.logged)
    for case in cases:
        history, prompts, candidates = [], set(), set()
        for index in range(len(STRATEGIES)):
            row = engine.query(case, index + 1, 1, (index, 0, 0), [],
                               prompts, candidates, conf["policy_seed"])
    from subprocess import run as process_run
    private_path = run / "pilot_private.json"
    process_run([sys.executable, "-m", "script.script_new.phase2_v3.private_score",
                 str(run / "pilot_queries.jsonl"), str(private_path)], cwd=ROOT, check=True)
    rows = read_jsonl(run / "pilot_queries.jsonl")
    public_candidates = sum(bool(x["public_feedback"]["candidate_values"]) for x in rows)
    private = json.loads(private_path.read_text())["aggregate"]
    signal = public_candidates > 0 or private["decoded_recovered_queries"] > 0
    result = {"target_calls": len(rows) + sanity["target_calls"],
              "pilot_target_calls": len(rows), "sanity": sanity,
              "target_runtime": target.metadata,
              "public_candidate_queries": public_candidates,
              "private_aggregate_file": str(private_path),
              "status": "SIGNAL_PRESENT" if signal else "NO_LEARNING_SIGNAL"}
    _cost_event(run, "pilot_queries", len(rows) - existing_calls)
    write_json(run / "pilot_summary.json", result)
    return result


def _save_checkpoint(run: Path, model, vec, base: ControllerEnv, calls: int) -> None:
    import numpy as np
    import torch
    step = model.num_timesteps
    stem = run / f"checkpoint_{step:04d}"
    model.save(str(stem) + ".zip")
    vec.save(str(stem) + ".norm.pkl")
    state = {"timesteps": step, "logged_target_calls": calls,
             "target_call_intents": len(base.engine.attempts),
             "budget_per_episode": base.budget,
             "episode": base.episode, "case_id": base.case["case_id"] if base.case else None,
             "step_index": base.step_index, "history": base.history,
             "seen_prompts": sorted(base.seen_prompts),
             "seen_candidates": sorted(base.seen_candidates), "used": sorted(base.used),
             "last_action": base.last_action, "last_feedback": base.last_feedback,
             "base_rng": base.rng.getstate(), "python_rng": random.getstate(),
             "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
             "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    with (Path(str(stem) + ".state.pkl")).open("wb") as f:
        pickle.dump(state, f)
    write_json(Path(str(stem) + ".manifest.json"),
               {"timesteps": step, "logged_target_calls": calls,
                "target_call_intents": len(base.engine.attempts),
                "runtime_versions": _runtime_versions(),
                "policy_sha256": sha256(Path(str(stem) + ".zip")),
                "normalizer_sha256": sha256(Path(str(stem) + ".norm.pkl")),
                "state_sha256": sha256(Path(str(stem) + ".state.pkl")),
                "protocol_sha256": sha256(run / "protocol.json")})


def _cost_event(run: Path, event: str, target_calls: int) -> None:
    from datetime import datetime, timezone
    with (run / "cost_events.jsonl").open("a") as f:
        f.write(json.dumps({"event": event, "target_calls": target_calls,
                            "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")


def _runtime_versions() -> dict:
    from importlib.metadata import version
    return {name: version(name) for name in ("stable-baselines3", "gymnasium", "torch",
                                             "transformers", "peft")}


def _metric_float(value):
    """Convert NumPy scalar metrics to JSON-compatible native floats."""
    return None if value is None else float(value)


def migrate(run: Path, dry: bool) -> dict:
    """Carry the completed fix2 PPO update into this code-only metrics repair."""
    _, frozen, split = _read_prepared(run)
    from script.script_new.phase2_v3_long_fix2.cli import _read_prepared as verify_source
    _, old, old_split = verify_source(MIGRATION_SOURCE_RUN)
    if split != old_split or frozen["config_sha256"] != old["config_sha256"]:
        raise ValueError("Migration changes the split or PPO configuration")
    changed = {name for name, value in old["protocol_hashes"].items()
               if frozen["protocol_hashes"].get(name) != value}
    if changed != {"v3:__init__.py", "v3:cli.py"}:
        raise ValueError(f"Migration changes model/renderer inputs: {sorted(changed)}")
    if frozen["source_pilot_hashes"] != old["source_pilot_hashes"]:
        raise ValueError("Migration changes source pilot")
    source = MIGRATION_SOURCE_RUN
    if (source / "train_summary.json").exists():
        raise ValueError("Source run is already complete")
    stem = "checkpoint_0512"
    source_manifest_path = source / f"{stem}.manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if (source_manifest["timesteps"], source_manifest["logged_target_calls"],
            source_manifest["target_call_intents"]) != (512, 512, 512):
        raise ValueError("Source checkpoint is not the complete first PPO rollout")
    if source_manifest["protocol_sha256"] != sha256(source / "protocol.json"):
        raise ValueError("Source checkpoint protocol hash mismatch")
    for suffix, field in ((".zip", "policy_sha256"), (".norm.pkl", "normalizer_sha256"),
                          (".state.pkl", "state_sha256")):
        if sha256(source / f"{stem}{suffix}") != source_manifest[field]:
            raise ValueError(f"Source checkpoint corrupt: {suffix}")
    queries = read_jsonl(source / "train_queries.jsonl")
    attempts = read_jsonl(source / "train_queries_target_attempts.jsonl")
    if len(queries) != 512 or len(attempts) != 512:
        raise ValueError("Source query or intent count differs from checkpoint")
    if {x["query_id"] for x in queries} != {x["query_id"] for x in attempts}:
        raise ValueError("Source contains unresolved target-call intents")
    image_dir = source / "train_queries_images"
    image_files = sorted(p.relative_to(source) for p in image_dir.rglob("*") if p.is_file())
    files = ["train_queries.jsonl", "train_queries_target_attempts.jsonl",
             "train_metrics.jsonl", "cost_events.jsonl",
             f"{stem}.zip", f"{stem}.norm.pkl", f"{stem}.state.pkl"]
    file_hashes = {name: sha256(source / name) for name in files}
    result = {"status": "ready_to_migrate" if dry else "migrated",
              "source_run": str(source), "destination_run": str(run),
              "preserved_target_queries": 512, "preserved_image_files": len(image_files),
              "source_protocol_sha256": sha256(source / "protocol.json"),
              "source_checkpoint_manifest_sha256": sha256(source_manifest_path),
              "copied_file_hashes": file_hashes}
    if dry:
        return result
    from shutil import copyfile

    def copy_exact(relative: str | Path) -> None:
        relative = Path(relative)
        destination = run / relative
        origin = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256(destination) != sha256(origin):
                raise FileExistsError(f"Migrated file differs: {destination}")
        else:
            copyfile(origin, destination)
            if sha256(destination) != sha256(origin):
                raise RuntimeError(f"Migrated file hash mismatch: {destination}")

    for name in files:
        copy_exact(name)
    for relative in image_files:
        copy_exact(relative)
    new_manifest = dict(source_manifest)
    new_manifest["protocol_sha256"] = sha256(run / "protocol.json")
    target_manifest_path = run / f"{stem}.manifest.json"
    if target_manifest_path.exists():
        if json.loads(target_manifest_path.read_text()) != new_manifest:
            raise FileExistsError("Migrated checkpoint manifest differs")
    else:
        write_json(target_manifest_path, new_manifest)
    result["new_checkpoint_manifest_sha256"] = sha256(target_manifest_path)
    migration_path = run / "checkpoint_migration.json"
    if migration_path.exists():
        if json.loads(migration_path.read_text()) != result:
            raise FileExistsError("Migration provenance differs")
    else:
        write_json(migration_path, result)
    return result


def train(run: Path, dry: bool, resume: bool) -> dict:
    conf, _, split = _read_prepared(run)
    if dry:
        return {"train_target_calls_max": conf["train_target_calls_max"],
                "checkpoint_interval": conf["checkpoint_interval"], "models_loaded": False,
                "gate": "pilot_summary.json must show SIGNAL_PRESENT"}
    pilot_summary = json.loads((run / "pilot_summary.json").read_text())
    if pilot_summary["status"] != "SIGNAL_PRESENT":
        raise RuntimeError("NO_LEARNING_SIGNAL: revise the repertoire on dev before PPO")
    if (run / "train_summary.json").exists():
        raise FileExistsError("Training summary exists; preserve the completed run")
    ensure_gpu()
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    target = _target()
    _verify_target_runtime(run, target)
    sanity = target.sanity()
    _cost_event(run, "train_sanity", sanity["target_calls"])
    engine = QueryEngine(target, "train", run / "train_queries.jsonl",
                         conf["max_attacker_retries"], conf["train_target_calls_max"])
    existing_calls = len(engine.logged)
    existing_intents = len(engine.attempts)
    if existing_intents and not resume:
        raise FileExistsError("Training log exists; pass --resume")
    base = ControllerEnv(_case_subset(split, "dev_train"), engine,
                         conf["train_episode_budget"], conf["policy_seed"])
    raw = DummyVecEnv([lambda: gym_env(base)])
    checkpoints = sorted(run.glob("checkpoint_*.manifest.json"))
    if resume and checkpoints:
        last = checkpoints[-1]
        meta = json.loads(last.read_text())
        if meta.get("runtime_versions") != _runtime_versions():
            raise ValueError("Resume runtime versions differ from checkpoint")
        stem = str(last).removesuffix(".manifest.json")
        if meta["protocol_sha256"] != sha256(run / "protocol.json"):
            raise ValueError("Checkpoint protocol mismatch")
        for suffix, field in ((".zip", "policy_sha256"), (".norm.pkl", "normalizer_sha256"),
                              (".state.pkl", "state_sha256")):
            if sha256(Path(stem + suffix)) != meta[field]:
                raise ValueError("Checkpoint hash mismatch")
        with Path(stem + ".state.pkl").open("rb") as f:
            state = pickle.load(f)
        if state["budget_per_episode"] != conf["train_episode_budget"]:
            raise ValueError("Checkpoint episode budget mismatch")
        vec = VecNormalize.load(stem + ".norm.pkl", raw)
        # The saved policy/normalizer are consistent at the rollout boundary,
        # but the fresh ControllerEnv is not restored mid-episode. Force a new
        # episode before the first resumed action.
        model = PPO.load(stem + ".zip", env=vec, device="cpu", force_reset=True)
        base.episode = max([state["episode"]] + [r["episode"] for r in engine.attempts.values()])
        base.rng.setstate(state["base_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("torch_cuda_rng"):
            torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
        lost_calls = existing_intents - meta["target_call_intents"]
        if lost_calls < 0:
            raise ValueError("Training log shorter than checkpoint")
    else:
        if resume and existing_intents:
            raise RuntimeError("No consistent checkpoint for resume; real calls preserved")
        random.seed(conf["policy_seed"])
        np.random.seed(conf["policy_seed"])
        torch.manual_seed(conf["policy_seed"])
        vec = VecNormalize(raw, norm_obs=True, norm_reward=False)
        model = PPO("MlpPolicy", vec, learning_rate=conf["ppo"]["learning_rate"],
                    n_steps=conf["checkpoint_interval"], batch_size=conf["ppo"]["batch_size"],
                    n_epochs=conf["ppo"]["n_epochs"], clip_range=conf["ppo"]["clip_range"],
                    ent_coef=conf["ppo"]["ent_coef"], seed=conf["policy_seed"],
                    device="cpu", verbose=1)
        lost_calls = 0

    class AuditCallback(BaseCallback):
        def _on_step(self):
            return True

        def _capture(self):
            _save_checkpoint(run, self.model, vec, base, len(engine.logged))
            from collections import Counter
            recent = list(engine.logged.values())[-conf["checkpoint_interval"]:]
            metrics = dict(self.model.logger.name_to_value)
            gradients = [p.grad.detach().float().norm().item() for p in self.model.policy.parameters()
                         if p.grad is not None]
            record = {"checkpoint": self.num_timesteps,
                      "entropy_loss": _metric_float(metrics.get("train/entropy_loss")),
                      "approx_kl": _metric_float(metrics.get("train/approx_kl")),
                      "clip_fraction": _metric_float(metrics.get("train/clip_fraction")),
                      "explained_variance": _metric_float(metrics.get("train/explained_variance")),
                      "gradient_norm": sum(x*x for x in gradients)**0.5 if gradients else None,
                      "strategy_distribution": dict(Counter(r["strategy_id"] for r in recent)),
                      "target_calls_by_modality": dict(Counter(r["requested_modality"] for r in recent))}
            with (run / "train_metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")

        def _on_rollout_start(self):
            checkpoint = run / f"checkpoint_{self.num_timesteps:04d}.manifest.json"
            if (self.num_timesteps and self.num_timesteps % conf["checkpoint_interval"] == 0
                    and not checkpoint.exists()):
                self._capture()

        def _on_training_end(self):
            self._capture()

    remaining = conf["train_target_calls_max"] - existing_intents
    if remaining <= 0:
        return {"status": "budget_exhausted", "target_call_intents": existing_intents,
                "successful_logged_queries": existing_calls}
    interval = conf["checkpoint_interval"]
    steps = (remaining // interval) * interval
    if steps == 0:
        raise RuntimeError("Fewer than one complete rollout remains; real calls preserved")
    model.learn(total_timesteps=steps, reset_num_timesteps=not resume,
                callback=AuditCallback())
    result = {"status": "trained", "target_call_intents": len(engine.attempts),
              "successful_logged_queries": len(engine.logged),
              "unresolved_call_intents": len(set(engine.attempts) - set(engine.logged)),
              "lost_unupdated_calls_on_resume": lost_calls,
              "sanity_target_calls": sanity["target_calls"],
              "runtime_versions": _runtime_versions(),
              "resume_type": "checkpoint_consistent; not exact last-step or GPU bitwise"}
    _cost_event(run, "train_queries_new", len(engine.logged) - existing_calls)
    write_json(run / "train_summary.json", result)
    return result


def _policy_action(model, vec, obs):
    import numpy as np
    norm = vec.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    action, _ = model.predict(norm, deterministic=True)
    return tuple(int(x) for x in action[0])


def validate(run: Path, dry: bool) -> dict:
    conf, _, split = _read_prepared(run)
    cases = _case_subset(split, "dev_validation")
    checkpoints = [run / f"checkpoint_{step:04d}.manifest.json"
                   for step in conf["checkpoints_for_selection"]]
    if dry:
        return {"methods": ["uniform", "fixed", "ppo"], "cases": len(cases),
                "budget_each": conf["validation_episode_budget"],
                "target_calls_per_method_per_checkpoint": 48,
                "checkpoints": [p.name for p in checkpoints], "models_loaded": False}
    missing = [p.name for p in checkpoints if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing predefined long-run checkpoints: {missing}")
    if (run / "validation_summary.json").exists():
        raise FileExistsError("Validation summary exists; preserve the completed comparison")
    ensure_gpu()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    train_summary = json.loads((run / "train_summary.json").read_text())
    if train_summary["successful_logged_queries"] != conf["train_target_calls_max"]:
        raise ValueError("Long training did not complete its frozen query budget")
    if train_summary["runtime_versions"] != _runtime_versions():
        raise ValueError("Validation runtime versions differ from training")
    pilot_labels = json.loads((run / "pilot_private.json").read_text())["labels"]
    pilot_rows = {r["query_id"]: r for r in read_jsonl(run / "pilot_queries.jsonl")}
    scores = {i: 0 for i in range(12)}
    for label in pilot_labels:
        if label["decoded_recovered"]:
            scores[next(j for j, s in enumerate(STRATEGIES)
                        if s.id == pilot_rows[label["query_id"]]["strategy_id"])] += 1
    fixed = min(scores, key=lambda i: (-scores[i], i))
    target = _target()
    _verify_target_runtime(run, target)
    sanity = target.sanity()
    _cost_event(run, "validation_sanity", sanity["target_calls"])
    summaries = []
    for checkpoint in checkpoints:
        step = int(checkpoint.stem.split("_")[1].split(".")[0])
        stem = str(checkpoint).removesuffix(".manifest.json")
        for method in ("uniform", "fixed", "ppo"):
            path = run / f"validation_{step}_{method}.jsonl"
            engine = QueryEngine(target, f"validation:{step}:{method}", path,
                                 conf["max_attacker_retries"])
            existing_validation_calls = len(engine.logged)
            base = ControllerEnv(cases, engine, conf["validation_episode_budget"],
                                 conf["policy_seed"])
            model = vec = None
            if method == "ppo":
                raw = DummyVecEnv([lambda: gym_env(base)])
                vec = VecNormalize.load(stem + ".norm.pkl", raw)
                vec.training = False
                model = PPO.load(stem + ".zip", device="cpu")
            for case in cases:
                obs, _ = base.reset(case=case)
                for t in range(conf["validation_episode_budget"]):
                    if method == "uniform":
                        action = ((t + conf["policy_seed"]) % 12, 0, 0)
                    elif method == "fixed":
                        action = (fixed, 0, 0)
                    else:
                        action = _policy_action(model, vec, obs)
                    obs, _, _, _, _ = base.step(action)
            summaries.append({"checkpoint": step, "method": method,
                              "target_calls": len(engine.logged), "trace": str(path)})
            from subprocess import run as process_run
            process_run([sys.executable, "-m", "script.script_new.phase2_v3.private_score",
                         str(path), str(run / f"validation_{step}_{method}_private.json")],
                        cwd=ROOT, check=True)
            labels = json.loads((run / f"validation_{step}_{method}_private.json").read_text())["labels"]
            label_by_id = {x["query_id"]: x for x in labels}
            trace_rows = read_jsonl(path)
            if len(trace_rows) != len(cases) * conf["validation_episode_budget"]:
                raise RuntimeError("Validation trace has the wrong paired budget")
            for case in cases:
                steps = [r["step"] for r in trace_rows if r["case_id"] == case["case_id"]]
                if sorted(steps) != list(range(1, conf["validation_episode_budget"] + 1)):
                    raise RuntimeError("Validation case missing or duplicated a query step")
            prefixes = {}
            for cutoff in conf["validation_prefixes"]:
                eligible = [r for r in trace_rows if r["step"] <= cutoff]
                prefixes[str(cutoff)] = {
                    "queries": len(eligible),
                    "decoded_recovered_cases": len({r["case_id"] for r in eligible
                                                    if label_by_id[r["query_id"]]["decoded_recovered"]}),
                    "exact_recovered_cases": len({r["case_id"] for r in eligible
                                                  if label_by_id[r["query_id"]]["exact_recovered"]})}
            summaries[-1]["prefix_metrics"] = prefixes
            _cost_event(run, f"validation_{step}_{method}", len(engine.logged) - existing_validation_calls)
    result = {"sanity_target_calls": sanity["target_calls"], "fixed_strategy": STRATEGIES[fixed].id,
              "runs": summaries,
              "total_validation_target_calls": sum(x["target_calls"] for x in summaries),
              "selection_rule": conf["selection"]}
    write_json(run / "validation_summary.json", result)
    return result


def freeze(run: Path, dry: bool) -> dict:
    conf, frozen, split = _read_prepared(run)
    if dry and not (run / "validation_summary.json").exists():
        return {"status": "awaiting_validation", "selection_rule": conf["selection"],
                "required": "validation_summary.json and four checkpoint/private-score pairs",
                "models_loaded": False}
    validation = json.loads((run / "validation_summary.json").read_text())
    ppo = [x for x in validation["runs"] if x["method"] == "ppo"]
    if sorted(x["checkpoint"] for x in ppo) != conf["checkpoints_for_selection"]:
        raise ValueError("Validation is missing a predefined PPO checkpoint")
    ranks = []
    for item in ppo:
        labels = json.loads((run / f"validation_{item['checkpoint']}_ppo_private.json").read_text())["labels"]
        first = {}
        for label in labels:
            if label["decoded_recovered"]:
                first.setdefault(label["case_id"],
                                 int(label["query_id"].rsplit(":", 1)[-1]))
        ranks.append((-len(first), sum(first.values()) if first else 10**9,
                      item["checkpoint"]))
    selected = min(ranks)[2]
    stem = run / f"checkpoint_{selected:04d}"
    checkpoint_manifest = json.loads(Path(str(stem) + ".manifest.json").read_text())
    if checkpoint_manifest["protocol_sha256"] != sha256(run / "protocol.json") or (
            checkpoint_manifest["policy_sha256"] != sha256(Path(str(stem) + ".zip")) or
            checkpoint_manifest["normalizer_sha256"] != sha256(Path(str(stem) + ".norm.pkl"))):
        raise ValueError("Selected checkpoint hash/protocol mismatch")
    if checkpoint_manifest.get("runtime_versions") != json.loads((run / "train_summary.json").read_text())["runtime_versions"]:
        raise ValueError("Checkpoint runtime differs from training")
    bundle = {"schema": "phase2-v3-freeze-1", "selected_checkpoint": selected,
              "selection_rule": conf["selection"], "config_sha256": sha256(HERE / "config.json"),
              "protocol_sha256": sha256(run / "protocol.json"),
              "split_sha256": sha256(run / "split.json"),
              "repertoire_sha256": sha256(run / "repertoire.json"),
              "policy_sha256": sha256(Path(str(stem) + ".zip")),
              "normalizer_sha256": sha256(Path(str(stem) + ".norm.pkl")),
              "decoder_sha256": frozen["protocol_hashes"]["canary_scoring.py"],
              "font_sha256": frozen["protocol_hashes"]["font"],
              "renderer_version": conf["renderer_version"],
              "target_runtime": json.loads((run / "pilot_summary.json").read_text()).get("target_runtime"),
              "training_runtime_versions": json.loads((run / "train_summary.json").read_text()).get("runtime_versions"),
              "observation_version": OBS_VERSION,
              "budget": conf["validation_episode_budget"],
              "known_exposed_case_ids": sorted(split["dev_train"] + split["dev_validation"] +
                                               json.loads((DATA / "manifest.json").read_text())["phase2_eval_case_ids"]),
              "limitation": "Unlogged prior exposures cannot be ruled out by software"}
    if not dry:
        path = run / "freeze.json"
        if path.exists() and json.loads(path.read_text()) != bundle:
            raise FileExistsError("Frozen artifact differs; create a new experiment")
        if not path.exists():
            write_json(path, bundle)
    return bundle


def evaluate(run: Path, manifest_path: Path, dry: bool) -> dict:
    _, frozen_protocol, _ = _read_prepared(run)
    bundle = json.loads((run / "freeze.json").read_text())
    for name, path in (("config_sha256", HERE / "config.json"),
                       ("protocol_sha256", run / "protocol.json"),
                       ("split_sha256", run / "split.json"),
                       ("repertoire_sha256", run / "repertoire.json")):
        if bundle[name] != sha256(path):
            raise ValueError(f"Frozen artifact mismatch: {name}")
    if bundle["decoder_sha256"] != frozen_protocol["protocol_hashes"]["canary_scoring.py"] or (
            bundle["font_sha256"] != frozen_protocol["protocol_hashes"]["font"]):
        raise ValueError("Frozen decoder/renderer mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "phase2-v3-evaluation-manifest-1":
        raise ValueError("Explicit evaluation manifest required")
    public = Path(manifest["public_cases"])
    adapter = Path(manifest["adapter"])
    if sha256(public) != manifest["public_cases_sha256"]:
        raise ValueError("Evaluation public cases hash mismatch")
    if sha256(adapter / "adapter_model.safetensors") != manifest["adapter_sha256"]:
        raise ValueError("Evaluation adapter hash mismatch")
    cases = read_jsonl(public)
    if any(set(c) != {"case_id", "prefix"} for c in cases):
        raise ValueError("Evaluation public cases must contain only case_id and prefix")
    ids = [c["case_id"] for c in cases]
    if any(digest(c["prefix"])[:12] != c["case_id"] for c in cases):
        raise ValueError("Evaluation case ID does not match public prefix")
    if len(set(ids)) != len(ids) or set(ids) & set(bundle["known_exposed_case_ids"]):
        raise ValueError("Evaluation identities overlap known train/dev/eval exposures")
    if manifest.get("previous_exposures") is None:
        raise ValueError("Evaluation manifest must record previous exposures")
    if set(ids) & set(manifest["previous_exposures"]):
        raise ValueError("Evaluation identities were previously exposed")
    if manifest["adapter_sha256"] == json.loads((run / "protocol.json").read_text())["protocol_hashes"]["adapter"]:
        raise ValueError("Confirmatory evaluation requires a newly prepared target")
    setup = Path(manifest["canary_setup_manifest"])
    if sha256(setup) != manifest["canary_setup_manifest_sha256"]:
        raise ValueError("New canary setup manifest hash mismatch")
    private_cases = Path(manifest["private_cases"])
    if sha256(private_cases) != manifest["private_cases_sha256"]:
        raise ValueError("Private evaluation cases hash mismatch")
    training = json.loads((adapter / "training_manifest.json").read_text())
    if training.get("status") != "complete" or training.get("adapter_sha256") != manifest["adapter_sha256"]:
        raise ValueError("New target training provenance mismatch")
    result = {"cases": len(cases), "budget": bundle["budget"],
              "planned_target_calls": len(cases) * bundle["budget"],
              "prior_exposures_recorded": len(manifest["previous_exposures"]),
              "unlogged_exposure_limit": bundle["limitation"]}
    if dry:
        return result
    ensure_gpu()
    if bundle.get("training_runtime_versions") != _runtime_versions():
        raise ValueError("Evaluation runtime versions differ from frozen policy")
    trace = run / "new_evaluation_queries.jsonl"
    if trace.exists():
        raise FileExistsError("Evaluation trace exists; preserve it")
    stem = run / f"checkpoint_{bundle['selected_checkpoint']:04d}"
    if sha256(Path(str(stem) + ".zip")) != bundle["policy_sha256"] or sha256(
            Path(str(stem) + ".norm.pkl")) != bundle["normalizer_sha256"]:
        raise ValueError("Frozen policy/normalizer mismatch")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    sys.path.insert(0, str(ROOT / "script/script_new"))
    target = HistoricalTarget(adapter)
    result["sanity"] = target.sanity()
    _cost_event(run, "new_evaluation_sanity", result["sanity"]["target_calls"])
    engine = QueryEngine(target, "new-evaluation", trace, config()["max_attacker_retries"])
    base = ControllerEnv(cases, engine, bundle["budget"], config()["policy_seed"])
    raw = DummyVecEnv([lambda: gym_env(base)])
    vec = VecNormalize.load(str(stem) + ".norm.pkl", raw)
    vec.training = False
    model = PPO.load(str(stem) + ".zip", device="cpu")
    for case in cases:
        obs, _ = base.reset(case=case)
        for _ in range(bundle["budget"]):
            obs, _, _, _, _ = base.step(_policy_action(model, vec, obs))
    result["actual_target_calls"] = len(engine.logged) + result["sanity"]["target_calls"]
    _cost_event(run, "new_evaluation_queries", len(engine.logged))
    result["trace_sha256"] = sha256(trace)
    from subprocess import run as process_run
    private_result = run / "new_evaluation_private.json"
    process_run([sys.executable, "-m", "script.script_new.phase2_v3.private_score",
                 str(trace), str(private_result), "--private-cases", str(private_cases)],
                cwd=ROOT, check=True)
    result["private_score_file"] = str(private_result)
    write_json(run / "new_evaluation_summary.json", result)
    return result


def summarize(run: Path) -> dict:
    result = {"run": str(run)}
    for name in ("pilot_summary", "train_summary", "validation_summary", "freeze",
                 "new_evaluation_summary"):
        path = run / f"{name}.json"
        result[name] = json.loads(path.read_text()) if path.exists() else None
    events = run / "cost_events.jsonl"
    if events.exists():
        rows = read_jsonl(events)
        result["cost_events"] = rows
        result["logged_target_calls_including_sanity"] = sum(r["target_calls"] for r in rows)
    attempts = list(run.glob("*_target_attempts.jsonl"))
    result["target_call_intents_by_trace"] = {p.name: len(read_jsonl(p)) for p in attempts}
    validation = result.get("validation_summary")
    if validation:
        grouped = {}
        for item in validation["runs"]:
            grouped.setdefault(item["checkpoint"], {})[item["method"]] = item
        result["rl_advantage_by_checkpoint"] = {
            str(step): (methods["ppo"]["prefix_metrics"]["12"]["decoded_recovered_cases"] >
                        max(methods[x]["prefix_metrics"]["12"]["decoded_recovered_cases"]
                            for x in ("uniform", "fixed")))
            for step, methods in grouped.items() if set(methods) == {"ppo", "uniform", "fixed"}}
        result["claim_limit"] = "Four validation cases are exploratory; one difference cannot establish an RL advantage."
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase II v3; GPU commands require CUDA_VISIBLE_DEVICES=0")
    parser.add_argument("command", choices=("diagnose", "prepare", "migrate", "probe", "pilot", "train",
                                             "validate", "freeze", "evaluate", "summarize"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="For probe only: count benign GPU target calls")
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args(argv)
    if a.command == "diagnose":
        path = a.output or a.run / "diagnosis"
        report = diagnose(path)
        result = {"json": str(path / "diagnosis.json"), "markdown": str(path / "diagnosis.md"),
                  "train_queries": report["train_queries"],
                  "integrity_pass": all(v["matches"] for v in report["integrity"].values())}
    elif a.command == "prepare":
        result = prepare(a.run, a.dry_run)
    elif a.command == "migrate":
        result = migrate(a.run, a.dry_run)
    elif a.command == "probe":
        result = probe(a.run, a.dry_run, a.gpu)
    elif a.command == "pilot":
        result = pilot(a.run, a.dry_run)
    elif a.command == "train":
        result = train(a.run, a.dry_run, a.resume)
    elif a.command == "validate":
        result = validate(a.run, a.dry_run)
    elif a.command == "freeze":
        result = freeze(a.run, a.dry_run)
    elif a.command == "evaluate":
        if not a.manifest:
            parser.error("evaluate requires --manifest")
        result = evaluate(a.run, a.manifest, a.dry_run)
    else:
        result = summarize(a.run)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
