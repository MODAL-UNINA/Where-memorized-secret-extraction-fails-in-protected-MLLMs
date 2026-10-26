"""Versioned stochastic Qwen2-VL generation without changing the frozen runner."""
from __future__ import annotations

from pathlib import Path


def generate(target, prompt: str, image: Path | None, decoding: dict, seed: int) -> str:
    """Return an unfiltered sample; the caller records the seed and filter view."""
    if (decoding.get("do_sample") is not True or decoding.get("use_cache") is not True
            or decoding.get("top_k") != 0
            or not 0 < decoding["temperature"] <= 2
            or not 0 < decoding["top_p"] <= 1 or decoding["max_new_tokens"] < 1
            or not 0 <= seed < 2**32):
        raise ValueError("invalid frozen sampling parameters")
    if target.family != "qwen2vl" or target.protected:
        raise ValueError("sampled ladder requires a raw Qwen2-VL target")
    torch = target.torch
    messages, pil_image = target._messages_and_image(prompt, image)
    text = target.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = target.processor(text=[text], images=[pil_image] if pil_image else None,
                              return_tensors="pt", return_mm_token_type_ids=True)
    if target.device == "cuda":
        inputs = inputs.to("cuda:0")
    torch.manual_seed(seed)
    if target.device == "cuda":
        torch.cuda.manual_seed_all(seed)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    with torch.inference_mode():
        output = target.model.generate(
            **inputs, max_new_tokens=decoding["max_new_tokens"], do_sample=True,
            temperature=decoding["temperature"], top_p=decoding["top_p"], top_k=0,
            use_cache=decoding["use_cache"],
        )
    if target.device == "cuda":
        end.record()
        torch.cuda.synchronize(0)
        target.last_gpu_seconds = start.elapsed_time(end) / 1000.0
    else:
        target.last_gpu_seconds = 0.0
    generated = output[:, inputs["input_ids"].shape[1]:]
    return target.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
