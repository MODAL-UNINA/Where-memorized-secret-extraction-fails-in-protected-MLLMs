#!/usr/bin/env python3
"""FigStep Qwen-dev-selected images on qualified LLaVA v3 eval seeds 101–103.

Reuses the exact E5d statement order, renderer, font and public image assets.
The only new target path is the LLaVA v3 handoff and trainer-matched renderer.
No LLaVA result selects or reorders an attack. A CPU preflight verifies that
the 760-square source reaches LLaVA as a 336-square tensor before GPU calls.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

import defense_decomposition_v1 as dd
import figstep_ladder_v1 as fs
import ladder_common as lc
import llava_v3_measurements as lv3
import matched_baselines_ladder_v1 as mb
import revision_v1_arms as arms
from protected_interface import filter_output

ROOT = dd.ROOT
PROTOCOL = "revision_v1-figstep-llava-v3-v1"
OUTPUT_ROOT = ROOT / "private/figstep_llava_v3"
OUTPUT = OUTPUT_ROOT / "eval"
MODEL = lv3.MODEL
TARGET_SEEDS = (101, 102, 103)
MAX_NEW_TOKENS = 64
CONDITIONS = dd.CONDITIONS
BUDGETS = fs.BUDGETS


def image_size_dict(value: object) -> dict:
    """Normalize transformers SizeDict and ordinary processor size mappings."""
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"unsupported image processor size type: {type(value).__name__}")


def seed_paths(seed: int) -> dict[str, Path]:
    if seed not in TARGET_SEEDS:
        raise ValueError("unsupported LLaVA v3 target seed")
    partition = 1000 + seed
    run = ROOT / f"runs/revision_v1_llava_v3_lr2e5/target_seed_{seed}"
    qualification = (ROOT / "private/llava_v3_lr2e5" if seed == 101 else
                     ROOT / f"private/llava_v3_lr2e5_seed{seed}")
    public = dd.target_paths(seed, partition)
    return {"run": run, "qualification": qualification,
            "adapter": run / "eval/adapter", "handoff": run / "eval_target_plan.json",
            "protocol": run / "protocol.json", "gate": qualification / "eval_gate.json",
            "dev_qualification": qualification / "dev_qualification.json",
            "queries": public["queries"], "partition_seed": partition}


def validate_target(seed: int) -> tuple[dict, list[dict]]:
    """Read only public cases and key-free LLaVA v3 qualification artifacts."""
    path = seed_paths(seed)
    handoff = json.loads(path["handoff"].read_text(encoding="utf-8"))
    expected = {"protocol": "revision_v1_llava_v3_lr2e5", "status": "validated",
                "split": "eval", "target_seed": seed,
                "partition_seed": path["partition_seed"], "family": "llava15",
                "model": MODEL, "gate_status": "PASS"}
    if any(handoff.get(name) != value for name, value in expected.items()):
        raise ValueError(f"LLaVA v3 handoff is not qualified for seed {seed}")
    if json.loads(path["dev_qualification"].read_text(encoding="utf-8")).get("status") != "PASS" or \
       json.loads(path["gate"].read_text(encoding="utf-8")).get("status") != "PASS":
        raise ValueError(f"LLaVA v3 dev/eval qualification failed for seed {seed}")
    protocol = json.loads(path["protocol"].read_text(encoding="utf-8"))
    if protocol.get("protocol") != handoff["protocol"] or \
       handoff.get("adapter") != dd.rel(path["adapter"]) or \
       handoff.get("adapter_sha256") != dd.sha(path["adapter"] / "adapter_model.safetensors") or \
       handoff.get("gate_sha256") != dd.sha(path["gate"]) or \
       handoff.get("queries_path") != dd.rel(path["queries"]) or \
       handoff.get("queries_sha256") != dd.sha(path["queries"]):
        raise ValueError(f"LLaVA v3 handoff artifacts changed for seed {seed}")
    cases = arms.public_cases(path["queries"], seed, path["partition_seed"])
    return {"target_seed": seed, "partition_seed": path["partition_seed"],
            "adapter": dd.rel(path["adapter"]),
            "adapter_sha256": handoff["adapter_sha256"],
            "target_plan_sha256": dd.sha(path["handoff"]),
            "target_protocol_sha256": dd.sha(path["protocol"]),
            "dev_qualification_sha256": dd.sha(path["dev_qualification"]),
            "eval_gate_sha256": dd.sha(path["gate"]),
            "public_cases_sha256": dd.sha(path["queries"]),
            "identities": [case["canary_id"] for case in cases]}, cases


def preflight_value() -> dict:
    """Use the actual LLaVA image processor to verify the square 336px path."""
    from PIL import Image
    from transformers import AutoProcessor
    from io import BytesIO

    sample = Image.open(BytesIO(fs.render(fs.fill(fs.statements()[0], {
        "record_ref": "User_eval_101_0000",
        "prefix": "Confidential identifier for User_eval_101_0000: ",
    })))).convert("RGB")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    image_processor = processor.image_processor
    pixels = image_processor(images=sample, return_tensors="pt")["pixel_values"]
    shape = list(pixels.shape)
    if sample.size != fs.CANVAS or shape[-2:] != [336, 336]:
        raise ValueError(f"LLaVA preprocessing is not the expected square 760→336 path: {shape}")
    size = image_size_dict(image_processor.size)
    crop = image_size_dict(image_processor.crop_size)
    if (size.get("shortest_edge") != 336 or
            crop.get("height") != 336 or
            crop.get("width") != 336 or not image_processor.do_resize or
            not image_processor.do_center_crop):
        raise ValueError(f"unexpected LLaVA resize/crop configuration: {size}, {crop}")
    return {"protocol": PROTOCOL, "model": MODEL,
            "source_size": list(sample.size), "pixel_values_shape": shape,
            "processor_class": type(image_processor).__name__, "resize_size": size,
            "crop_size": crop, "do_center_crop": bool(image_processor.do_center_crop),
            "geometric_text_crop": False,
            "figstep_renderer_sha256": dd.sha(Path(fs.__file__)),
            "llava_renderer_sha256": dd.sha(Path(__file__).with_name("llava_revision_target_v2.py"))}


def preflight() -> dict:
    value = preflight_value()
    path = OUTPUT_ROOT / "preflight.json"
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("LLaVA preprocessing changed after preflight")
    else:
        lc.write_private(path, value)
    return value


def plan(write_assets: bool = False) -> dict:
    preflight_path = OUTPUT_ROOT / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("run the CPU preflight before planning LLaVA FigStep")
    value = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (value.get("protocol"), value.get("source_size"), value.get("pixel_values_shape", [])[-2:]) != (
            PROTOCOL, list(fs.CANVAS), [336, 336]) or value.get("geometric_text_crop") is not False:
        raise ValueError("invalid LLaVA image preprocessing preflight")
    if value.get("figstep_renderer_sha256") != dd.sha(Path(fs.__file__)) or \
       value.get("llava_renderer_sha256") != dd.sha(Path(__file__).with_name("llava_revision_target_v2.py")):
        raise ValueError("renderer changed after LLaVA image preprocessing preflight")
    statements = fs.statements()
    order, selection_sha = fs.load_selection(statements)
    targets, calls = [], []
    for seed in TARGET_SEEDS:
        target, cases = validate_target(seed)
        targets.append(target)
        for system in dd.SYSTEMS:
            for rank, statement_index in enumerate(order, start=1):
                for case in cases:
                    statement = fs.fill(statements[statement_index], case)
                    image, image_sha = fs.asset("eval", seed, case["canary_id"],
                                                statement_index, statement, write_assets)
                    calls.append({"target": "implanted", "target_seed": seed,
                                  "partition_seed": target["partition_seed"],
                                  "canary_id": case["canary_id"], "system": system,
                                  "method": fs.METHOD, "template_index": statement_index,
                                  "rank": rank, "statement": statement,
                                  "prompt": fs.TEXT_PROMPT, "image": image,
                                  "image_sha256": image_sha})
    expected = len(TARGET_SEEDS) * 13 * len(dd.SYSTEMS) * len(statements)
    if len(calls) != expected:
        raise ValueError(f"expected {expected} FigStep calls, planned {len(calls)}")
    here = Path(__file__)
    return {"protocol": PROTOCOL, "split": "eval", "scope": "LLaVA v3 FigStep on three qualified targets; "
            "statement order transferred from Qwen dev with no LLaVA tuning",
            "model": MODEL, "family": "llava15", "decoding": {"do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS}, "methods": [fs.METHOD], "budgets": list(BUDGETS),
            "order": order, "statements": statements, "text_prompt": fs.TEXT_PROMPT,
            "qwen_dev_selection_sha256": selection_sha,
            "figstep_renderer_sha256": dd.sha(Path(fs.__file__)),
            "figstep_font_sha256": dd.sha(fs.FONT),
            "preflight_sha256": dd.sha(preflight_path),
            "systems_sha256": {name: dd.text_sha(value) for name, value in dd.SYSTEMS.items()},
            "conditions": {name: list(pair) for name, pair in CONDITIONS.items()},
            "filter_sha256": dd.sha(here.with_name("protected_interface.py")),
            "llava_renderer_sha256": dd.sha(here.with_name("llava_revision_target_v2.py")),
            "scorer_sha256": dd.sha(here.with_name("private_scorer.py")),
            "script_sha256": dd.sha(here), "targets": targets, "calls": calls}


def run(args: argparse.Namespace) -> None:
    from gpu_runtime import configure_gpu
    from llava_revision_target_v2 import LlavaRevisionTargetV2

    plan_path, intents_path, responses_path = dd.files(args.output)
    if args.resume:
        if not all(path.is_file() for path in (plan_path, intents_path, responses_path)):
            raise ValueError("resume requires original plan and both ledgers")
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if frozen != plan():
            raise ValueError("frozen LLaVA FigStep plan or artifacts changed")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"preserve existing run: {args.output}")
        args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(args.output, 0o700)
        frozen = plan()
        plan_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        os.chmod(plan_path, 0o600)
        for path in (intents_path, responses_path):
            path.touch(mode=0o600)
    intents, responses = dd.ledger(args.output, frozen)
    if len(intents) != len(responses):
        raise RuntimeError("an intent has no response; the LLaVA call may have occurred, so do not retry")
    if len(responses) == len(frozen["calls"]):
        print(f"Already complete: {len(responses)} calls")
        return
    configure_gpu(args.gpu_id, args.disable_cudnn)
    adapters = {item["target_seed"]: str(ROOT / item["adapter"]) for item in frozen["targets"]}
    target, loaded = None, None
    with intents_path.open("a", encoding="utf-8") as intent_file, \
         responses_path.open("a", encoding="utf-8") as response_file:
        for index in range(len(responses), len(frozen["calls"])):
            call = frozen["calls"][index]
            if call["target_seed"] != loaded:
                if target is not None:
                    import gc
                    import torch
                    del target
                    gc.collect()
                    torch.cuda.empty_cache()
                target = LlavaRevisionTargetV2("llava15", MODEL, adapters[call["target_seed"]],
                                               "cuda", protected=False)
                loaded = call["target_seed"]
                print(f"Loaded LLaVA v3 target {loaded}", flush=True)
            target.system = dd.SYSTEMS[call["system"]]
            image = ROOT / call["image"]
            if dd.sha(image) != call["image_sha256"]:
                raise ValueError("FigStep image changed after plan")
            dd.append_durable(intent_file, {"index": index, "call": call, "started_at": time.time()})
            started = time.perf_counter()
            raw = target.generate(call["prompt"], image, MAX_NEW_TOKENS)
            dd.append_durable(response_file, {"index": index, "raw": raw,
                                              "filtered": filter_output(raw),
                                              "wall_seconds": time.perf_counter() - started,
                                              "gpu_seconds": None})
            if (index + 1) % 25 == 0 or index + 1 == len(frozen["calls"]):
                print(f"LLaVA v3 FigStep calls complete: {index + 1}/{len(frozen['calls'])}", flush=True)


def report(args: argparse.Namespace) -> dict:
    frozen, responses = lc.complete_ledger(args.output, plan)
    keys = mb.read_keys("eval", frozen)
    for call in frozen["calls"]:
        secret = keys[(call["target_seed"], call["canary_id"])].lower()
        if secret in call["prompt"].lower() or secret in call["statement"].lower():
            raise ValueError("private secret appears in a FigStep target input")
    summary = fs.summarize("eval", frozen["calls"], responses, keys)
    summary["cost"][fs.METHOD]["wall_seconds"] = sum(float(row.get("wall_seconds") or 0)
                                                     for row in responses)
    plan_path, intents_path, responses_path = dd.files(args.output)
    result = {"protocol": PROTOCOL, "split": "eval", "scope": frozen["scope"],
              "total_calls": len(responses), **summary,
              "preflight_sha256": frozen["preflight_sha256"],
              "qwen_dev_selection_sha256": frozen["qwen_dev_selection_sha256"],
              "plan_sha256": dd.sha(plan_path), "intents_sha256": dd.sha(intents_path),
              "responses_sha256": dd.sha(responses_path)}
    lc.write_private(args.output / "private_report.json", result)
    for row in summary["curves"]:
        if row["budget"] == 12:
            print(f"FigStep LLaVA v3 {row['condition']:26s} {row['recovered']}/{row['identities']} "
                  f"by_seed={row['recovered_by_target_seed']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("plan")
    lc.add_run_parsers(sub, OUTPUT)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2))
    elif args.command == "plan":
        frozen = plan(write_assets=True)
        print(json.dumps({"protocol": PROTOCOL, "calls": len(frozen["calls"]),
                          "target_seeds": list(TARGET_SEEDS),
                          "qwen_dev_selection_sha256": frozen["qwen_dev_selection_sha256"],
                          "preflight_sha256": frozen["preflight_sha256"]}, indent=2))
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
