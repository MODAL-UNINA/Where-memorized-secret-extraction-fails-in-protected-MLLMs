#!/usr/bin/env python3
"""JSON-lines Qwen2.5 attacker worker, isolated on its own visible GPU.

The parent sends only public case fields and attacker-observable feedback.
Evaluator keys and LoRA adapters are never opened here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        trust_remote_code=False,
    ).eval()
    print(json.dumps({"ready": True, "model": args.model}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        messages = [{"role": "system", "content": request["system"]},
                    {"role": "user", "content": request["user"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
        torch.manual_seed(int(request["seed"]))
        torch.cuda.manual_seed_all(int(request["seed"]))
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.95, use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(json.dumps({"text": answer.strip(), "wall_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
