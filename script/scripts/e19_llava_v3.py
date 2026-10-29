#!/usr/bin/env python3
"""E19-LLaVA: E19's two unseen neutral instructions on the LLaVA v3 eval targets 101-103.

E11-LLaVA (``e11_llava_v3.py``) unchanged except for the instructions: the E5e
calls (Direct, DAN, Paraphrase; 12 templates each in the frozen order, 39
identities) and the 100 synthetic utility tasks are sent under
``e19_neutral_context_v1.SYSTEMS`` instead of E11's wordings (3 x 2 x 568 =
3,408 calls), rendered with ``LlavaRevisionTargetV2``, greedy, 64 new tokens.
The plan additionally pins the E19 preregistration. ``report`` is E11-LLaVA's:
rows for neutral and v1_protected (E5e ledgers) and both E19 wordings, exact
McNemar against ``v1_protected`` with Holm within each view; the paired test
against ``neutral`` is not part of the E11-LLaVA code path, and the neutral
counts are reported as rows.
"""
from __future__ import annotations

from pathlib import Path

import defense_decomposition_v1 as dd
import e11_llava_v3 as base
import e19_neutral_context_v1 as e19

PROTOCOL = "e19-llava-v3-v1"
base.PROTOCOL = PROTOCOL
base.OUTPUT = base.ROOT / "private/e19_llava_v3_v1"
base.SYSTEMS = dict(e19.SYSTEMS)
_base_plan_for = base.plan_for


def plan_for() -> dict:
    e19.validate_systems()
    prereg = e19.preregistration()
    here = Path(__file__)
    return {**_base_plan_for(), "protocol": PROTOCOL, "preregistration": prereg,
            "e19_script_sha256": dd.sha(here.with_name("e19_neutral_context_v1.py")),
            "e19_llava_script_sha256": dd.sha(here)}


base.plan_for = plan_for


if __name__ == "__main__":
    base.main()
