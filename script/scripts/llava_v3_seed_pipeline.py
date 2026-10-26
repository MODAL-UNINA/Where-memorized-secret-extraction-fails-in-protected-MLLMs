#!/usr/bin/env python3
"""E8b: the full LLaVA v3 measurement chain for an additional target seed.

Applies, unchanged, the stages and rules used for target 101
(``llava_v3_measurements.py``) to target seed 102 or 103, then the fixed
attacks measured on 101:

1. dev gate, dev utility, dev qualification (gate PASS and strict neutral
   utility >= 33/100, the floor fixed on 18 October);
2. only after a dev PASS: eval gate and eval utility;
3. only after an eval gate PASS: defense decomposition (six fixed prompts x
   two systems), E5e fixed baselines (Qwen dev order) and E9 two-turn
   re-encoding.

The stage code is reused by pointing the 101 modules at this seed's paths;
every output goes to a seed-specific directory:
``private/llava_v3_lr2e5_seed<S>/``, ``private/utility_llava_v3_seed<S>_<split>/``,
``private/defense_decomposition_llava_v3_seed<S>/``,
``private/matched_baselines_llava_v3_seed<S>/``,
``private/twoturn_reencoding_v1/llava_v3_seed<S>/``. Every stage is skipped
when its report exists, so the command is resumable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import defense_decomposition_v1 as dd
import llava_v3_measurements as lv3

ROOT = dd.ROOT


def configure(seed: int) -> None:
    if seed == 101:
        raise ValueError("target 101 has its own frozen outputs; use llava_v3_measurements.py")
    partition = 1000 + seed
    lv3.TARGET_SEED, lv3.PARTITION_SEED = seed, partition
    lv3.RUN = ROOT / f"runs/revision_v1_llava_v3_lr2e5/target_seed_{seed}"
    lv3.OUT = ROOT / f"private/llava_v3_lr2e5_seed{seed}"
    lv3.DATA = ROOT / f"private/revision_v1/target_seed_{seed}/partition_seed_{partition}"
    lv3.PUBLIC = ROOT / f"TDSC/revision_v1/target_seed_{seed}/partition_seed_{partition}"


def utility_dir(seed: int, split: str) -> Path:
    return ROOT / f"private/utility_llava_v3_seed{seed}_{split}"


def utility(seed: int, split: str, gpu_id: int) -> None:
    import utility_cost_v1 as uc

    if split == "eval":
        lv3.require_dev_pass()
        if lv3.gate_status("eval") is None:
            raise RuntimeError("run the eval gate before eval utility")
    target = {"name": f"llava_v3_seed{seed}_{split}", "family": "llava15", "model": lv3.MODEL,
              "adapter": dd.rel(lv3.adapter_dir(split)), "adapter_sha256": lv3.check_adapter(split),
              "partition_seed": lv3.PARTITION_SEED}
    uc.targets = lambda: [target]
    uc.PROTOCOL = f"revision_v1-llava-v3-lr2e5-seed{seed}-utility-{split}-v1"
    output = utility_dir(seed, split)
    if not (output / "report.json").exists():
        resume = (output / "plan.json").is_file()
        uc.run(argparse.Namespace(output=output, gpu_id=gpu_id, disable_cudnn=True, resume=resume))
        uc.report(argparse.Namespace(output=output))
    for cell in lv3.read_json(output / "report.json")["cells"]:
        if cell["view"] == "raw":
            print(f"{cell['target']} {cell['system']}: strict {cell['strict_correct']}/{cell['tasks']}")


def qualify_dev(seed: int) -> str:
    path = lv3.OUT / "dev_qualification.json"
    if not path.exists():
        status = lv3.gate_status("dev")
        if status is None:
            raise RuntimeError("run the dev gate first")
        report = utility_dir(seed, "dev") / "report.json"
        neutral = next(c["strict_correct"] for c in lv3.read_json(report)["cells"]
                       if c["view"] == "raw" and c["system"] == "neutral")
        checks = {"memorization_gate": status == "PASS", "utility_floor": neutral >= lv3.UTILITY_FLOOR}
        lv3.write_once(path, {"protocol": "revision_v1_llava_v3_lr2e5", "split": "dev", "target_seed": seed,
                              "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                              "utility_neutral_strict": neutral, "utility_floor": lv3.UTILITY_FLOOR,
                              "gate_sha256": lv3.sha(lv3.gate_path("dev")),
                              "utility_report_sha256": lv3.sha(report),
                              "adapter_sha256": lv3.check_adapter("dev")})
    value = lv3.read_json(path)
    print(f"target {seed} dev qualification: {value['status']} {value['checks']}")
    return value["status"]


def run_ledger(module_run, module_report, output: Path, **extra) -> None:
    if (output / "private_report.json").exists():
        print(f"Already reported: {output / 'private_report.json'}")
        return
    resume = (output / "plan.json").is_file()
    module_run(argparse.Namespace(output=output, resume=resume, disable_cudnn=True, **extra))
    module_report(argparse.Namespace(output=output, **{k: v for k, v in extra.items() if k != "gpu_id"}))


def measurements(seed: int, gpu_id: int) -> None:
    lv3.handoff()
    llava_dd = lv3.decomposition_module()
    llava_dd.TARGET_SEED, llava_dd.PARTITION_SEED = seed, lv3.PARTITION_SEED
    llava_dd.PROTOCOL = f"revision_v1-llava-v3-lr2e5-seed{seed}-defense-decomposition-v1"
    output = ROOT / f"private/defense_decomposition_llava_v3_seed{seed}"
    if not (output / "private_report.json").exists():
        resume = (output / "plan.json").is_file()
        llava_dd.run(argparse.Namespace(output=output, gpu_id=gpu_id, disable_cudnn=True, resume=resume))
        llava_dd.report(argparse.Namespace(output=output))

    import matched_baselines_llava_v3 as e5e
    e5e.TARGET_SEED, e5e.PARTITION_SEED = seed, lv3.PARTITION_SEED
    run_ledger(e5e.run, e5e.report, ROOT / f"private/matched_baselines_llava_v3_seed{seed}", gpu_id=gpu_id)

    import twoturn_reencoding_v1 as e9
    run_ledger(e9.run, e9.report, e9.OUTPUT_ROOT / f"llava_v3_seed{seed}", gpu_id=gpu_id, arch="llava_v3")


def main() -> None:
    import os
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-seed", type=int, choices=(102, 103), required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    seed = args.target_seed
    configure(seed)
    lv3.gate(argparse.Namespace(split="dev", gpu_id=args.gpu_id))
    utility(seed, "dev", args.gpu_id)
    if qualify_dev(seed) != "PASS":
        print(f"Target {seed}: dev qualification FAIL; eval measurements not run.")
        return
    lv3.gate(argparse.Namespace(split="eval", gpu_id=args.gpu_id))
    utility(seed, "eval", args.gpu_id)
    if lv3.gate_status("eval") != "PASS":
        print(f"Target {seed}: eval gate FAIL; decomposition and attacks not run.")
        return
    measurements(seed, args.gpu_id)
    print(f"Target {seed}: LLaVA v3 pipeline complete")


if __name__ == "__main__":
    main()
