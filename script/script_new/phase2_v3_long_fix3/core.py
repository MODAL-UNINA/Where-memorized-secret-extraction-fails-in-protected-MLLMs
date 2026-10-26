"""Small, model-free protocol primitives."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
HISTORICAL = ROOT / "runs/disjoint_lora_seed42/benign_replay_v2/phase2_ppo_seed42_retry1"
DATA = ROOT / "Dataset/disjoint_lora_seed42"
FONT = HERE / "assets/DejaVuSansMono.ttf"
CODE_FILES = ("target_attacker_phase2.py", "canary_scoring.py", "RL_phase2_corrected.py",
              "RL_phase2_disjoint.py", "callbacks_phase2.py", "audit_protected_interface.py",
              "integrity_check_phase2.py", "paper_target_recipe_v2.py")
V3_FILES = ("__init__.py", "__main__.py", "core.py", "strategies.py", "feedback.py",
            "runtime.py", "private_score.py", "diagnose.py", "cli.py")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def ensure_gpu() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 explicitly for GPU commands")


def public_cases() -> list[dict]:
    path = DATA / "dev_attack_public.jsonl"
    probe = json.loads((DATA / "full_text_probe_manifest.json").read_text())
    if sha256(path) != probe["dev"]["attack_cases_sha256"]:
        raise ValueError("dev public cases hash mismatch")
    rows = read_jsonl(path)
    if len(rows) != 12 or any(set(c) != {"case_id", "prefix"} for c in rows):
        raise ValueError("Unexpected dev public case contract")
    return rows


def protocol_hashes(adapter: Path | None = None) -> dict:
    paths = {name: ROOT / "script/script_new" / name for name in CODE_FILES}
    paths.update({f"v3:{name}": HERE / name for name in V3_FILES})
    paths.update({"dev_public": DATA / "dev_attack_public.jsonl",
                  "dev_dataset": DATA / "dev.jsonl",
                  "dev_private_probe": DATA / "dev_full_text_probe.jsonl",
                  "dev_recipe_dataset": DATA / "benign_replay_v2/dev.jsonl",
                  "dev_memorization_report": ROOT / "runs/disjoint_lora_seed42/benign_replay_v2/dev_full_text_probe.json",
                  "split_manifest": DATA / "manifest.json",
                  "probe_manifest": DATA / "full_text_probe_manifest.json",
                  "font": FONT})
    if adapter is not None:
        paths["adapter"] = Path(adapter) / "adapter_model.safetensors"
        paths["adapter_training_manifest"] = Path(adapter) / "training_manifest.json"
    return {key: sha256(path) for key, path in paths.items()}


def verify_hashes(expected: dict, adapter: Path | None = None) -> None:
    actual = protocol_hashes(adapter)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Protocol hash mismatch: {key}")


def load_split(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if obj.get("schema") != "phase2-v3-split-1":
        raise ValueError("Wrong split schema")
    cases = public_cases()
    ids = {c["case_id"] for c in cases}
    train, val = obj["dev_train"], obj["dev_validation"]
    if len(train) != 8 or len(val) != 4 or set(train) & set(val) or set(train + val) != ids:
        raise ValueError("Invalid dev split")
    if obj["source_sha256"] != sha256(DATA / "dev_attack_public.jsonl"):
        raise ValueError("Split source hash mismatch")
    return obj


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def no_placeholders(text: str) -> bool:
    return not bool(re.search(r"\{[^{}]+\}|\[[^\]]*(?:insert|placeholder|situation|target)[^\]]*\]", text, re.I))
