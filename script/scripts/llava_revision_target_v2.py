"""LLaVA revision_v1_v2 target with the trainer's exact prompt renderer.

The generic VisionTarget uses the model chat template; that template omits the
system role in LLaVA 1.5. This class uses the versioned trainer renderer for
both qualification scores and protected attack queries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finetune_llava15_v2 import assistant_label_start, render
from protected_interface import filter_output
from target_runner import VisionTarget

NEUTRAL_SYSTEM = "You are a helpful assistant."


class LlavaRevisionTargetV2(VisionTarget):
    def __init__(self, family: str, model_id: str, adapter: str | None, device: str,
                 system: str | None = None, *, protected: bool = True):
        if family != "llava15":
            raise ValueError("LlavaRevisionTargetV2 requires family llava15")
        if not protected and system not in (None, NEUTRAL_SYSTEM):
            raise ValueError("unprotected qualification requires the neutral training system")
        super().__init__(family, model_id, adapter, device,
                         system=system if protected else NEUTRAL_SYSTEM,
                         protected=protected)

    @staticmethod
    def _image(image: Path | None):
        if image is None:
            return None
        from PIL import Image
        with Image.open(image) as source:
            return source.convert("RGB")

    def _prefix(self, prompt: str, image: Path | None) -> str:
        return render(self.system, prompt, None, image is not None)

    def _inputs(self, prompt: str, image: Path | None, candidate: str | None = None):
        prefix = self._prefix(prompt, image)
        text = prefix if candidate is None else render(self.system, prompt, candidate,
                                                       image is not None)
        return self.processor(text=text, images=self._image(image), return_tensors="pt")

    def generate(self, prompt: str, image: Path | None, max_new_tokens: int) -> str:
        inputs = self._inputs(prompt, image)
        if self.device == "cuda":
            inputs = inputs.to("cuda:0")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False, use_cache=True)
        generated = output[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return filter_output(answer) if self.protected else answer

    def tokenization_hash(self, prompt: str, image: Path | None,
                          candidates: list[str]) -> str:
        prefix_ids = self._inputs(prompt, image)["input_ids"][0].tolist()
        full_ids = [self._inputs(prompt, image, candidate)["input_ids"][0].tolist()
                    for candidate in candidates]
        payload = {"renderer": "finetune_llava15_v2.render", "system": self.system,
                   "prefix": self._prefix(prompt, image), "prefix_ids": prefix_ids,
                   "candidate_full_ids": full_ids}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def token_logprobs(self, prompt: str, image: Path | None, candidate: str) -> list[float]:
        """Score contextual answer tokens, excluding EOS as in other rank probes."""
        prefix = self._inputs(prompt, image)
        full = self._inputs(prompt, image, candidate)
        ids = full["input_ids"][0]
        start = assistant_label_start(ids.tolist(), prefix["input_ids"][0].tolist())
        if self.device == "cuda":
            full = full.to("cuda:0")
        with self.torch.inference_mode():
            result = self.model(**full)
            logits = result.logits[:, start - 1:ids.shape[0] - 1, :].float()
            logprobs = self.torch.log_softmax(logits, dim=-1)
            selected = logprobs.gather(-1, full["input_ids"][:, start:].unsqueeze(-1))
        return [float(value) for value in selected[0, :, 0].tolist()]
