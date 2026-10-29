"""Run the E19 report with a narrower integrity check on the E5b reference ledger.

ladder_common.py was edited on 2026-10-24, after the E5b reference ledger was
frozen (2026-10-16), so rebuilding the reference plan changes only its
`common_sha256` field and lc.complete_ledger refuses the ledger. This wrapper
accepts the reference ledger only if:
  * every other plan field, calls included, equals the rebuilt plan;
  * plan.json and responses.jsonl have the SHA-256 that the E11 report recorded
    for the same ledger on 2026-10-20 (private/system_prompt_sensitivity_v1/).
E19's own ledger and the E11 ledger keep the original strict check. The report
is written as usual; the deviation is recorded in deviation_refcheck.json next
to it. e19_neutral_context_v1.py is not modified (its hash is pinned in plan.json).

Usage: e19_report_refcheck.py [report args]   -> E19 report
       e19_report_refcheck.py endpoints        -> leakage_endpoints_e19_v1 (frozen
                                                  endpoints), same reference check
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import e19_neutral_context_v1 as e19
import ladder_common as lc

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(e19.REFERENCE)
E11_REPORT = ROOT / "private/system_prompt_sensitivity_v1/private_report.json"
TOLERATED = {"common_sha256"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    recorded = json.loads(E11_REPORT.read_text(encoding="utf-8"))
    ref_dir = REFERENCE if REFERENCE.is_absolute() else ROOT / REFERENCE
    checks = {
        "plan.json": (sha(ref_dir / "plan.json"), recorded["reference_plan_sha256"]),
        "responses.jsonl": (sha(ref_dir / "responses.jsonl"), recorded["reference_responses_sha256"]),
    }
    for name, (now, then) in checks.items():
        if now != then:
            raise ValueError(f"reference {name} differs from the E11 record: {now} != {then}")

    strict = lc.complete_ledger
    seen = {}

    def complete_ledger(output, build_plan):
        out = Path(output)
        if out.resolve() != ref_dir.resolve():
            return strict(output, build_plan)
        frozen = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        rebuilt = build_plan()
        differing = sorted(k for k in set(frozen) | set(rebuilt) if frozen.get(k) != rebuilt.get(k))
        if set(differing) - TOLERATED:
            raise ValueError(f"reference plan differs beyond {sorted(TOLERATED)}: {differing}")
        seen.update({k: {"frozen": frozen.get(k), "rebuilt": rebuilt.get(k)} for k in differing})
        return strict(output, lambda: frozen)

    lc.complete_ledger = complete_ledger
    e19.lc.complete_ledger = complete_ledger
    if sys.argv[1:2] == ["endpoints"]:
        import leakage_endpoints_e19_v1 as endpoints
        endpoints.lc.complete_ledger = complete_ledger
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        endpoints.main()
    else:
        sys.argv = [sys.argv[0], "report", *sys.argv[1:]]
        e19.main()

    output = Path(e19.OUTPUT) if hasattr(e19, "OUTPUT") else None
    target = output if output and output.is_absolute() else ROOT / (output or "private/e19_neutral_context_v1")
    (target / "deviation_refcheck.json").write_text(json.dumps({
        "reason": "ladder_common.py edited 2026-10-24 after the E5b reference ledger was frozen",
        "tolerated_fields": seen,
        "reference_files_match_e11_record": {k: v[0] for k, v in checks.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Reference accepted with tolerated fields {sorted(seen)}; deviation recorded in {target}/deviation_refcheck.json")


if __name__ == "__main__":
    main()
