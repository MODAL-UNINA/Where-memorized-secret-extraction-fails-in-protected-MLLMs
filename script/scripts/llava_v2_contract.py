"""Key-free handoff validation for qualified LLaVA revision_v1_v2 targets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_target(plan_path: Path, adapter: Path, protocol_path: Path, *,
                    split: str, target_seed: int, partition_seed: int,
                    model: str) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = {"protocol": "revision_v1_llava_v2", "status": "validated",
                "split": split, "target_seed": target_seed,
                "partition_seed": partition_seed, "family": "llava15",
                "model": model, "gate_status": "PASS"}
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"LLaVA handoff differs: {field}")
    if Path(plan["adapter"]).resolve() != adapter.resolve():
        raise ValueError("LLaVA adapter path differs from qualified handoff")
    if plan["adapter_sha256"] != digest(adapter / "adapter_model.safetensors"):
        raise ValueError("LLaVA adapter weights changed after qualification")
    if plan["target_protocol_sha256"] != digest(protocol_path):
        raise ValueError("LLaVA frozen target protocol changed")
    if plan["queries_sha256"] != digest(Path(plan["queries_path"])):
        raise ValueError("LLaVA public queries changed")
    gate_path = Path(plan["gate_path"])
    if plan["gate_sha256"] != digest(gate_path):
        raise ValueError("LLaVA qualification report changed")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (gate.get("protocol"), gate.get("gate_mode"), gate.get("status"),
            gate.get("target_seeds"), gate.get("splits")) != (
            "revision_v1", "qualification", "PASS", [target_seed], [split]):
        raise ValueError("LLaVA target memorization gate has not passed")
    key = f"{target_seed}/{split}/llava15/{model}"
    provenance = gate.get("provenance", {}).get(key, {})
    if provenance.get("adapter_sha256") != plan["adapter_sha256"]:
        raise ValueError("LLaVA gate and adapter provenance differ")
    manifest_path = adapter.parent / "run_manifest.json"
    if plan["run_manifest_sha256"] != digest(manifest_path):
        raise ValueError("LLaVA adapter training manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("protocol") != "revision_v1_llava_v2" or
            manifest.get("target_seed") != target_seed or
            manifest.get("partition_seed") != partition_seed or
            manifest.get("model") != model or
            manifest.get("dataset_sha256") != provenance.get("training_sha256")):
        raise ValueError("LLaVA gate and adapter training differ")
    if plan["trainer_sha256"] != digest(Path(__file__).with_name("finetune_llava15_v2.py")):
        raise ValueError("LLaVA target trainer changed")
    if manifest.get("trainer_sha256") != plan["trainer_sha256"]:
        raise ValueError("LLaVA adapter was trained with another trainer")
    if plan["target_runtime_sha256"] != digest(Path(__file__).with_name("llava_revision_target_v2.py")):
        raise ValueError("LLaVA target renderer changed")
    if plan["phase2_target_sha256"] != digest(Path(__file__).with_name("llava_v2_phase2_target.py")):
        raise ValueError("LLaVA Phase-II target instrumentation changed")
    if plan["protected_interface_sha256"] != digest(Path(__file__).with_name("protected_interface.py")):
        raise ValueError("LLaVA protected interface changed")
    return plan
