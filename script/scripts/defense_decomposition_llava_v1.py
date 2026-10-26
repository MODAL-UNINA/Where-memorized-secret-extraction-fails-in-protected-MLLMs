#!/usr/bin/env python3
"""Defense-layer decomposition on the qualified LLaVA-1.5 v2 eval target.

Same design as ``defense_decomposition_v1.py`` (Qwen2-VL), applied to the
single qualified LLaVA target (target seed 101, partition 1101): the 13 public
eval identities, the same six fixed prompts, ``neutral`` and ``protected``
system instructions, raw responses stored privately and the frozen output
filter applied offline. The base LLaVA model answers the training prefix
under both systems as a memorization control.

Prompts are rendered with the LLaVA trainer's renderer through
``LlavaRevisionTargetV2``, as in qualification and Phase II. ``plan`` and
``run`` never open evaluator keys; ``report`` alone reads them.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import defense_decomposition_v1 as dd
import revision_v1_arms as arms
from llava_v2_contract import validate_target
from protected_interface import filter_output

ROOT = dd.ROOT
PROTOCOL = "revision_v1-llava-v2-defense-decomposition-v1"
OUTPUT = ROOT / "private/defense_decomposition_llava_v1"
MODEL = "llava-hf/llava-1.5-7b-hf"
TARGET_SEED, PARTITION_SEED = 101, 1101
LLAVA_TARGET = ROOT / "runs/revision_v1_llava_v2/target_seed_101"
LLAVA_RUN = LLAVA_TARGET / "phase2_v1"
MAX_NEW_TOKENS = 64


def plan(asset_dir: Path | None = None) -> dict:
    """Frozen call plan from public artifacts and the key-free LLaVA handoff."""
    owned_temporary = None
    if asset_dir is None:
        owned_temporary = tempfile.TemporaryDirectory()
        asset_dir = Path(owned_temporary.name)
    asset_dir = asset_dir.resolve()
    try:
        target_plan_path = LLAVA_RUN / "eval_target_plan.json"
        adapter = LLAVA_TARGET / "eval/adapter"
        target_protocol = LLAVA_RUN / "protocol.json"
        handoff = validate_target(target_plan_path, adapter, target_protocol, split="eval",
                                  target_seed=TARGET_SEED, partition_seed=PARTITION_SEED,
                                  model=MODEL)
        paths = dd.target_paths(TARGET_SEED, PARTITION_SEED)
        cases = arms.public_cases(paths["queries"], TARGET_SEED, PARTITION_SEED)
        if dd.sha(paths["queries"]) != handoff["queries_sha256"]:
            raise ValueError("LLaVA handoff and public eval cases differ")
        cue_images = arms.prepare_images(cases, asset_dir / f"target_seed_{TARGET_SEED}")
        implanted_calls, base_calls = [], []
        for system in dd.SYSTEMS:
            for prompt_id in dd.PROMPTS:
                for case in cases:
                    prompt, image, source = dd.prompt_input(case, prompt_id, paths["queries"].parent,
                                                            cue_images[case["canary_id"]])
                    call = {"target": "implanted", "target_seed": TARGET_SEED,
                            "partition_seed": PARTITION_SEED, "canary_id": case["canary_id"],
                            "system": system, "prompt_id": prompt_id, "prompt": prompt,
                            "image_source": source,
                            "image": (dd.rel(image) if source == "dataset" else
                                      str(image.relative_to(asset_dir)) if image else None),
                            "image_sha256": dd.sha(image) if image else None}
                    implanted_calls.append(call)
                    if prompt_id in dd.BASE_PROMPTS:
                        base_calls.append({**call, "target": "base"})
        calls = base_calls + implanted_calls
        expected = 13 * len(dd.SYSTEMS) * (len(dd.PROMPTS) + len(dd.BASE_PROMPTS))
        if len(calls) != expected:
            raise ValueError(f"expected {expected} calls, planned {len(calls)}")
        here = Path(__file__)
        return {"protocol": PROTOCOL,
                "scope": "post hoc defense-layer decomposition on the qualified LLaVA v2 eval target; "
                         "fixed prompts only, no tuning on eval",
                "model": MODEL, "family": "llava15",
                "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
                "systems_sha256": {name: dd.text_sha(value) for name, value in dd.SYSTEMS.items()},
                "prompts": list(dd.PROMPTS), "base_prompts": list(dd.BASE_PROMPTS),
                "conditions": {name: list(value) for name, value in dd.CONDITIONS.items()},
                "filter_sha256": dd.sha(here.with_name("protected_interface.py")),
                "renderer_sha256": dd.sha(here.with_name("llava_revision_target_v2.py")),
                "llava_trainer_sha256": dd.sha(here.with_name("finetune_llava15_v2.py")),
                "qwen_decomposition_sha256": dd.sha(here.with_name("defense_decomposition_v1.py")),
                "arms_sha256": dd.sha(here.with_name("revision_v1_arms.py")),
                "scorer_sha256": dd.sha(here.with_name("private_scorer.py")),
                "script_sha256": dd.sha(here),
                "private_key_boundary": "unisolated: keys readable on host but never opened by plan/run",
                "targets": [{"target_seed": TARGET_SEED, "partition_seed": PARTITION_SEED,
                             "adapter": dd.rel(adapter), "adapter_sha256": handoff["adapter_sha256"],
                             "target_plan_sha256": dd.sha(target_plan_path),
                             "target_protocol_sha256": dd.sha(target_protocol),
                             "public_cases_sha256": dd.sha(paths["queries"]),
                             "identities": [case["canary_id"] for case in cases]}],
                "calls": calls}
    finally:
        if owned_temporary is not None:
            owned_temporary.cleanup()


def run(args: argparse.Namespace) -> None:
    from gpu_runtime import configure_gpu
    from llava_revision_target_v2 import LlavaRevisionTargetV2

    plan_path, intents_path, responses_path = dd.files(args.output)
    asset_dir = args.output / "public_images"
    if args.resume:
        if not all(path.is_file() for path in (plan_path, intents_path, responses_path)):
            raise ValueError("resume requires original plan and both ledgers")
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan(asset_dir) != frozen:
            raise ValueError("frozen plan or target artifacts changed")
    else:
        # A run that failed before writing its plan made no target call.
        if args.output.exists() and any(path.name != "public_images" for path in args.output.iterdir()):
            raise FileExistsError(f"preserve existing run: {args.output}")
        args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(args.output, 0o700)
        frozen = plan(asset_dir)
        plan_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        os.chmod(plan_path, 0o600)
        for path in (intents_path, responses_path):
            path.touch(mode=0o600)
    intents, responses = dd.ledger(args.output, frozen)
    if len(intents) != len(responses):
        raise RuntimeError("an intent has no response; the target call may have occurred, so do not retry")
    if len(responses) == len(frozen["calls"]):
        print(f"Already complete: {len(responses)} calls")
        return
    configure_gpu(args.gpu_id, args.disable_cudnn)
    adapter = str(ROOT / frozen["targets"][0]["adapter"])
    target, loaded = None, object()
    with intents_path.open("a", encoding="utf-8") as intent_file, \
         responses_path.open("a", encoding="utf-8") as response_file:
        for index in range(len(responses), len(frozen["calls"])):
            call = frozen["calls"][index]
            wanted = call["target"]
            if wanted != loaded:
                if target is not None:
                    import gc
                    import torch
                    del target
                    gc.collect()
                    torch.cuda.empty_cache()
                # protected=False returns raw text; the system is set per call below.
                target = LlavaRevisionTargetV2("llava15", MODEL, None if wanted == "base" else adapter,
                                               "cuda", protected=False)
                loaded = wanted
                print(f"Loaded LLaVA target {wanted}", flush=True)
            target.system = dd.SYSTEMS[call["system"]]
            image = dd.resolve_image(call, asset_dir)
            dd.append_durable(intent_file, {"index": index, "call": call, "started_at": time.time()})
            started = time.perf_counter()
            raw = target.generate(call["prompt"], image, MAX_NEW_TOKENS)
            dd.append_durable(response_file, {"index": index, "raw": raw, "filtered": filter_output(raw),
                                             "wall_seconds": time.perf_counter() - started,
                                             "gpu_seconds": None})
            if (index + 1) % 25 == 0 or index + 1 == len(frozen["calls"]):
                print(f"LLaVA defense decomposition calls complete: {index + 1}/{len(frozen['calls'])}",
                      flush=True)


def report(args: argparse.Namespace) -> dict:
    plan_path, intents_path, responses_path = dd.files(args.output)
    frozen = json.loads(plan_path.read_text(encoding="utf-8"))
    if frozen != plan(args.output / "public_images"):
        raise ValueError("frozen plan or target artifacts changed")
    intents, responses = dd.ledger(args.output, frozen)
    if len(intents) != len(responses) or len(responses) != len(frozen["calls"]):
        raise ValueError("LLaVA defense decomposition calls are incomplete")
    paths = dd.target_paths(TARGET_SEED, PARTITION_SEED)
    manifest = json.loads(paths["data_manifest"].read_text(encoding="utf-8"))
    if manifest["sha256"]["eval_all_keys"] != dd.sha(paths["keys"]):
        raise ValueError("private key file differs from frozen data manifest")
    rows = {row["canary_id"]: row["secret"] for row in dd.read_jsonl(paths["keys"]) if row["kind"] == "text"}
    keys = {(TARGET_SEED, canary_id): rows[canary_id] for canary_id in frozen["targets"][0]["identities"]}
    for call in frozen["calls"]:
        if keys[(call["target_seed"], call["canary_id"])].lower() in call["prompt"].lower():
            raise ValueError("private secret appears in a target prompt")
    result = {"protocol": PROTOCOL, "scope": frozen["scope"], "total_calls": len(responses),
              **dd.summarize(frozen["calls"], responses, keys),
              "plan_sha256": dd.sha(plan_path), "intents_sha256": dd.sha(intents_path),
              "responses_sha256": dd.sha(responses_path), "keys_sha256": dd.sha(paths["keys"])}
    destination = args.output / "private_report.json"
    with destination.open("x", encoding="utf-8") as stream:
        os.chmod(destination, 0o600)
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    for row in result["conditions"]:
        if row["prompt_id"] == "any_prompt" or row["target"] == "base":
            print(f"{row['target']:9s} {row['condition']:26s} {row['prompt_id']:18s} "
                  f"{row['recovered']}/{row['identities']}")
    return result


def main() -> None:
    # The LLaVA handoff stores repository-relative paths.
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    runner = sub.add_parser("run")
    runner.add_argument("--output", type=Path, default=OUTPUT)
    runner.add_argument("--gpu-id", type=int, required=True)
    runner.add_argument("--disable-cudnn", action="store_true")
    runner.add_argument("--resume", action="store_true")
    scorer = sub.add_parser("report")
    scorer.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "plan":
        frozen = plan()
        counts: dict[str, int] = {}
        for call in frozen["calls"]:
            name = f"{call['target']}/{call['system']}/{call['prompt_id']}"
            counts[name] = counts.get(name, 0) + 1
        print(json.dumps({"protocol": PROTOCOL, "calls": len(frozen["calls"]),
                          "calls_by_cell": counts,
                          "adapter_sha256": frozen["targets"][0]["adapter_sha256"],
                          "systems_sha256": frozen["systems_sha256"]}, indent=2))
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
