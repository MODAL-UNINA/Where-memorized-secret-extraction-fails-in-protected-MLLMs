"""Shared provenance checks that prevent mixing target arms, splits, or runs."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


def validate_adapter(plan_path: Path, adapter: Path, arm: str, *, split: str,
                     family: str, model: str, seed: int) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "validated" or plan.get("split") != split:
        raise ValueError(f"target plan is not a validated {split} plan")
    if arm not in plan.get("arms", []): raise ValueError(f"target plan does not include the {arm} arm")
    if int(plan.get("seed", -1)) != seed or plan.get("family") != family or plan.get("model") != model:
        raise ValueError("target plan seed/family/model differs from this run")
    planned_adapter = Path(plan["output"]) / arm / "adapter"
    if planned_adapter.resolve() != adapter.resolve():
        raise ValueError(f"adapter is not the {arm} adapter declared by the target plan")
    if not (adapter / "adapter_config.json").is_file(): raise FileNotFoundError(f"{arm} adapter is incomplete")
    if plan.get("adapter_sha256") is not None:
        actual = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        if actual != plan["adapter_sha256"]:
            raise ValueError("adapter weights differ from the frozen target plan")
    run_manifest_path = adapter.parent / "run_manifest.json"
    if not run_manifest_path.is_file(): raise FileNotFoundError(f"missing trainer manifest: {run_manifest_path}")
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if int(run.get("seed", -1)) != seed or run.get("model") != model:
        raise ValueError("trainer manifest seed/model differs from the frozen target plan")
    if run.get("dataset_sha256") != plan.get("datasets", {}).get(f"{arm}_sha256"):
        raise ValueError("trained adapter dataset hash differs from the frozen target plan")
    runtime = plan.get("runtime")
    if runtime is not None:
        if run.get("gpu_id") != runtime.get("gpu_id") or run.get("disable_cudnn") != runtime.get("disable_cudnn"):
            raise ValueError("trainer GPU/cuDNN settings differ from the frozen target plan")
    mapping = {"epochs": "epochs", "max_steps": "max_steps", "batch_size": "batch_size",
               "grad_accumulation": "grad_accumulation", "learning_rate": "learning_rate",
               "lr_scheduler_type": "lr_scheduler_type", "warmup_ratio": "warmup_ratio",
               "dataloader_workers": "dataloader_workers",
               "dataloader_prefetch_factor": "dataloader_prefetch_factor",
               "checkpoint_strategy": "checkpoint_strategy", "save_total_limit": "save_total_limit"}
    for plan_name, run_name in mapping.items():
        if plan_name not in plan.get("hyperparameters", {}) or run_name not in run:
            raise ValueError(f"missing frozen trainer hyperparameter {run_name}")
        if run.get(run_name) != plan.get("hyperparameters", {}).get(plan_name):
            raise ValueError(f"trainer hyperparameter {run_name} differs from the frozen target plan")
    return plan


def validate_target_protocol(protocol_path: Path, plan_path: Path, role: str) -> dict:
    if role not in {"dev", "eval"}: raise ValueError("role must be dev or eval")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN": raise ValueError("target protocol is not frozen")
    expected_path = Path(protocol[f"{role}_plan"])
    if expected_path.resolve() != plan_path.resolve(): raise ValueError(f"{role} plan path differs from frozen protocol")
    actual_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if protocol.get(f"{role}_plan_sha256") != actual_hash: raise ValueError(f"{role} plan changed after protocol freeze")
    return protocol
