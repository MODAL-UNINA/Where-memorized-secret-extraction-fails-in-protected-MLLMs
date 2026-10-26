"""Qualified LLaVA v2 target with Phase-II GPU timing instrumentation."""
from __future__ import annotations

from pathlib import Path

from llava_revision_target_v2 import LlavaRevisionTargetV2


class LlavaV2Phase2Target(LlavaRevisionTargetV2):
    def generate(self, prompt: str, image: Path | None, max_new_tokens: int) -> str:
        if self.device != "cuda":
            self.last_gpu_seconds = 0.0
            return super().generate(prompt, image, max_new_tokens)
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        response = super().generate(prompt, image, max_new_tokens)
        end.record()
        self.torch.cuda.synchronize(0)
        self.last_gpu_seconds = start.elapsed_time(end) / 1000.0
        return response
