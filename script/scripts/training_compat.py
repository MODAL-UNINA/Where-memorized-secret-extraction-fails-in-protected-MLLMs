"""Compatibility helpers for Transformers TrainingArguments API changes."""
from __future__ import annotations

import math


def calculate_warmup_steps(
    *, record_count: int, batch_size: int, gradient_accumulation_steps: int,
    epochs: float, max_steps: int, warmup_ratio: float,
) -> tuple[int, int]:
    """Return planned optimizer steps and warmup steps, respectively.

    Recent Transformers releases expose ``warmup_steps`` but not
    ``warmup_ratio``. This computes the ratio in optimizer-step units.
    ``max_steps`` takes precedence over epochs, as it does in Trainer.
    """
    if record_count < 1 or batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("record count, batch size and gradient accumulation must be positive")
    if epochs <= 0 or max_steps == 0 or max_steps < -1:
        raise ValueError("epochs must be positive and max_steps must be -1 or positive")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")

    dataloader_batches = math.ceil(record_count / batch_size)
    updates_per_epoch = max(dataloader_batches // gradient_accumulation_steps, 1)
    planned_steps = max_steps if max_steps > 0 else math.ceil(epochs * updates_per_epoch)
    return planned_steps, math.ceil(planned_steps * warmup_ratio)
