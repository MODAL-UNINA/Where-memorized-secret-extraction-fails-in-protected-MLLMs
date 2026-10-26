#!/usr/bin/env python3
"""E5c-neutral: frozen revision_v1 MM-LEAK policies with the neutral system.

Completes the MM-LEAK row of the 4-level ladder. E5c
(``evaluate_policy_system_only_v1.py``) ran the frozen policies with the
confidentiality system prompt, giving ``system_prompt_only`` (raw) and
``system_prompt_and_filter`` (filtered). This run changes only the system
prompt to the neutral implantation instruction (``defense_decomposition_v1``
``SYSTEMS["neutral"]``), giving ``no_defense`` (raw) and ``filter_only``
(filtered) from the same trajectories.

Everything else is the E5c code path, imported unchanged so that the E5c
script and its frozen plan stay byte-identical: same five policies (first
seed per target), same 65 eval identities, budget 10, raw response as
attacker feedback, same filter, scorer and resume rules. Out-of-condition
evaluation, no retraining or tuning.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import defense_decomposition_v1 as dd
import evaluate_policy_system_only_v1 as base

base.SECURITY_SYSTEM_PROMPT = dd.SYSTEMS["neutral"]
base.PROTOCOL = "revision_v1-neutral-policy-eval-v1"
base.OUTPUT = base.ROOT / "private/neutral_policy_eval_v1"
base.CONDITIONS = {"no_defense": "raw", "filter_only": "filtered"}

_e5c_plan = base.plan


def plan() -> dict:
    frozen = _e5c_plan()
    frozen["scope"] = "frozen revision_v1 policies, out-of-condition with the neutral system; no retraining or tuning"
    frozen["system"] = "neutral"
    frozen["wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return frozen


base.plan = plan

if __name__ == "__main__":
    base.main()
