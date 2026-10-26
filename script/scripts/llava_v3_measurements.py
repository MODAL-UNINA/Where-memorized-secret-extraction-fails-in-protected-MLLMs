#!/usr/bin/env python3
"""E8 stages 2-3: qualify and measure the LLaVA v3 (learning rate 2e-5) adapters.

Stages, each refusing to start before its prerequisite:

1. ``gate --split dev``: paired memorization probe of base vs v3 dev adapter
   (``verify_memorization_llava_v2.py``) and the frozen gate report
   (``report_memorization.py``), as for the v2 adapters.
2. ``utility --split dev``: the 100 synthetic utility tasks of
   ``utility_cost_v1`` under the neutral and protected system prompts.
3. ``qualify-dev``: PASS only if the dev gate passes AND the dev adapter's
   strict neutral utility is at least ``UTILITY_FLOOR``. The floor is the
   LLaVA base neutral score (43/100) minus 10 and was fixed on 18 October 2026,
   before any v3 result existed.
4. Only after a dev PASS: ``gate --split eval``, ``utility --split eval`` and
   ``decomposition`` (the six fixed prompts x two systems of
   ``defense_decomposition_llava_v1`` on the v3 eval adapter).

No hyperparameter or checkpoint is chosen from eval results. Outputs are
private: ``private/llava_v3_lr2e5/``, ``private/utility_llava_v3_<split>/``,
``private/defense_decomposition_llava_v3/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import defense_decomposition_v1 as dd

ROOT = dd.ROOT
SCRIPTS = Path(__file__).resolve().parent
RUN = ROOT / "runs/revision_v1_llava_v3_lr2e5/target_seed_101"
OUT = ROOT / "private/llava_v3_lr2e5"
DATA = ROOT / "private/revision_v1/target_seed_101/partition_seed_1101"
PUBLIC = ROOT / "TDSC/revision_v1/target_seed_101/partition_seed_1101"
MODEL = "llava-hf/llava-1.5-7b-hf"
TARGET_SEED, PARTITION_SEED = 101, 1101
LEARNING_RATE = 2e-5
UTILITY_FLOOR = 33  # LLaVA base neutral strict 43/100 minus 10; fixed 2026-10-18 before v3 results


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def adapter_dir(split: str) -> Path:
    return RUN / split / "adapter"


def check_adapter(split: str) -> str:
    """Validate the trained adapter against the recorded recipe; return its weight hash."""
    recipe = read_json(RUN / "recipe.json")
    if recipe.get("changed", {}).get("learning_rate", {}).get("to") != LEARNING_RATE:
        raise ValueError("recipe does not declare learning rate 2e-5")
    weights = adapter_dir(split) / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"{split} adapter not trained yet: {weights}")
    manifest = read_json(RUN / split / "run_manifest.json")
    training = DATA / f"{split}_train_relative.jsonl"
    if (manifest.get("learning_rate") != LEARNING_RATE or manifest.get("seed") != TARGET_SEED
            or manifest.get("dataset_sha256") != sha(training)
            or not manifest["dataset_sha256"].startswith(recipe["dataset_sha256_prefix"][split])
            or manifest.get("trainer_sha256") != recipe["trainer_sha256"]):
        raise ValueError(f"{split} run manifest differs from the recorded v3 recipe")
    return sha(weights)


def gate_path(split: str) -> Path:
    return OUT / f"{split}_gate.json"


def gate_status(split: str) -> str | None:
    path = gate_path(split)
    return read_json(path).get("status") if path.is_file() else None


def require_dev_pass() -> dict:
    path = OUT / "dev_qualification.json"
    if not path.is_file() or read_json(path).get("status") != "PASS":
        raise RuntimeError("eval-adapter measurements require dev_qualification.json with status PASS")
    return read_json(path)


def gate(args: argparse.Namespace) -> None:
    split = args.split
    if split == "eval":
        require_dev_pass()
    check_adapter(split)
    OUT.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(OUT, 0o700)
    probe = OUT / f"{split}_probe.jsonl"
    if not probe.exists():
        subprocess.run([sys.executable, "-u", str(SCRIPTS / "verify_memorization_llava_v2.py"),
                        "--keys", str(DATA / f"{split}_all_keys.jsonl"),
                        "--negative-keys", str(DATA / f"{split}_negative_keys.jsonl"),
                        "--training", str(DATA / f"{split}_train_relative.jsonl"),
                        "--image-root", str(PUBLIC), "--negative-image-root", str(DATA),
                        "--adapter", str(adapter_dir(split)), "--output", str(probe),
                        "--target-seed", str(TARGET_SEED), "--split", split,
                        "--gpu-id", str(args.gpu_id), "--disable-cudnn"], check=True)
        os.chmod(probe, 0o600)
    if not gate_path(split).exists():
        # report_memorization exits 2 on FAIL after writing the report; both outcomes are recorded.
        result = subprocess.run([sys.executable, "-u", str(SCRIPTS / "report_memorization.py"),
                                 "--logs", str(probe), "--output", str(gate_path(split))])
        if result.returncode not in (0, 2):
            raise RuntimeError(f"report_memorization failed with code {result.returncode}")
    print(f"{split} gate: {gate_status(split)}")


def utility_module(split: str):
    import utility_cost_v1 as uc

    weights_sha = check_adapter(split)
    target = {"name": f"llava_v3_{split}", "family": "llava15", "model": MODEL,
              "adapter": dd.rel(adapter_dir(split)), "adapter_sha256": weights_sha,
              "partition_seed": PARTITION_SEED}
    uc.targets = lambda: [target]
    uc.PROTOCOL = f"revision_v1-llava-v3-lr2e5-utility-{split}-v1"
    return uc


def utility(args: argparse.Namespace) -> None:
    split = args.split
    if split == "eval":
        require_dev_pass()
        if gate_status("eval") is None:
            raise RuntimeError("run the eval gate before eval utility")
    uc = utility_module(split)
    output = ROOT / f"private/utility_llava_v3_{split}"
    if not (output / "report.json").exists():
        resume = (output / "plan.json").is_file()
        uc.run(argparse.Namespace(output=output, gpu_id=args.gpu_id, disable_cudnn=True, resume=resume))
        uc.report(argparse.Namespace(output=output))
    for cell in read_json(output / "report.json")["cells"]:
        if cell["view"] == "raw":
            print(f"{cell['target']} {cell['system']}: strict {cell['strict_correct']}/{cell['tasks']}")


def utility_score(split: str, system: str) -> int:
    report = read_json(ROOT / f"private/utility_llava_v3_{split}/report.json")
    return next(c["strict_correct"] for c in report["cells"] if c["view"] == "raw" and c["system"] == system)


def qualify_dev(args: argparse.Namespace) -> None:
    path = OUT / "dev_qualification.json"
    if path.exists():
        print(read_json(path)["status"])
        return
    status = gate_status("dev")
    if status is None:
        raise RuntimeError("run the dev gate first")
    neutral = utility_score("dev", "neutral")
    checks = {"memorization_gate": status == "PASS", "utility_floor": neutral >= UTILITY_FLOOR}
    write_once(path, {"protocol": "revision_v1_llava_v3_lr2e5", "split": "dev",
                      "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                      "utility_neutral_strict": neutral, "utility_floor": UTILITY_FLOOR,
                      "gate_sha256": sha(gate_path("dev")),
                      "utility_report_sha256": sha(ROOT / "private/utility_llava_v3_dev/report.json"),
                      "adapter_sha256": check_adapter("dev")})
    print(read_json(path)["status"], checks)


def handoff() -> Path:
    """Key-free v3 eval handoff consumed by the decomposition, written once after an eval PASS."""
    qualification = require_dev_pass()
    if gate_status("eval") != "PASS":
        raise RuntimeError("decomposition requires an eval gate PASS")
    plan_path, protocol_path = RUN / "eval_target_plan.json", RUN / "protocol.json"
    if not protocol_path.exists():
        write_once(protocol_path, {"protocol": "revision_v1_llava_v3_lr2e5",
                                   "recipe_sha256": sha(RUN / "recipe.json"),
                                   "dev_qualification": qualification})
    if not plan_path.exists():
        queries = PUBLIC / "eval_queries.jsonl"
        write_once(plan_path, {"protocol": "revision_v1_llava_v3_lr2e5", "status": "validated", "split": "eval",
                               "target_seed": TARGET_SEED, "partition_seed": PARTITION_SEED,
                               "family": "llava15", "model": MODEL, "gate_status": "PASS",
                               "adapter": dd.rel(adapter_dir("eval")), "adapter_sha256": check_adapter("eval"),
                               "gate_sha256": sha(gate_path("eval")),
                               "queries_path": dd.rel(queries), "queries_sha256": sha(queries)})
    return plan_path


def decomposition_module():
    import defense_decomposition_llava_v1 as llava_dd

    def validate_target(plan_path, adapter, protocol_path, *, split, target_seed, partition_seed, model):
        plan = read_json(plan_path)
        expected = {"protocol": "revision_v1_llava_v3_lr2e5", "status": "validated", "split": split,
                    "target_seed": target_seed, "partition_seed": partition_seed, "model": model,
                    "gate_status": "PASS"}
        if any(plan.get(k) != v for k, v in expected.items()):
            raise ValueError("v3 handoff differs from the requested target")
        if Path(adapter).resolve() != (ROOT / plan["adapter"]).resolve() or \
                plan["adapter_sha256"] != sha(Path(adapter) / "adapter_model.safetensors"):
            raise ValueError("v3 adapter differs from the handoff")
        if plan["gate_sha256"] != sha(gate_path(split)) or read_json(protocol_path)["protocol"] != plan["protocol"]:
            raise ValueError("v3 gate or protocol changed after handoff")
        return plan

    llava_dd.LLAVA_TARGET = RUN
    llava_dd.LLAVA_RUN = RUN
    llava_dd.validate_target = validate_target
    llava_dd.PROTOCOL = "revision_v1-llava-v3-lr2e5-defense-decomposition-v1"
    return llava_dd


def decomposition(args: argparse.Namespace) -> None:
    handoff()
    llava_dd = decomposition_module()
    output = ROOT / "private/defense_decomposition_llava_v3"
    if not (output / "private_report.json").exists():
        resume = (output / "plan.json").is_file()
        llava_dd.run(argparse.Namespace(output=output, gpu_id=args.gpu_id, disable_cudnn=True, resume=resume))
        llava_dd.report(argparse.Namespace(output=output))
    else:
        print(f"Already reported: {output / 'private_report.json'}")


def status(_: argparse.Namespace) -> None:
    for split in ("dev", "eval"):
        trained = (adapter_dir(split) / "adapter_model.safetensors").is_file()
        util = (ROOT / f"private/utility_llava_v3_{split}/report.json").is_file()
        print(f"{split}: trained={trained} gate={gate_status(split)} utility_report={util}")
    path = OUT / "dev_qualification.json"
    print("dev qualification:", read_json(path)["status"] if path.is_file() else None)
    print("decomposition report:", (ROOT / "private/defense_decomposition_llava_v3/private_report.json").is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("gate", "utility"):
        stage = sub.add_parser(name)
        stage.add_argument("--split", choices=("dev", "eval"), required=True)
        stage.add_argument("--gpu-id", type=int, required=True)
    sub.add_parser("qualify-dev")
    stage = sub.add_parser("decomposition")
    stage.add_argument("--gpu-id", type=int, required=True)
    sub.add_parser("status")
    args = parser.parse_args()
    {"gate": gate, "utility": utility, "qualify-dev": qualify_dev,
     "decomposition": decomposition, "status": status}[args.command](args)


if __name__ == "__main__":
    main()
