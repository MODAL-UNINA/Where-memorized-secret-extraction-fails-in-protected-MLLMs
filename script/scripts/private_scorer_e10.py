"""Evaluator-private literal and per-element recovery scorer for E10."""
from __future__ import annotations

from e10_secret import literal_match, parse_secret, present_fields


def score_case(response: str, secret: str) -> dict:
    elements = parse_secret(secret)
    found = present_fields(response, elements)
    literal = literal_match(response, secret)
    count = sum(found.values())
    return {"decoded": count == 4, "literal": literal,
            "elements_present": found, "element_count": count,
            "all_elements": count == 4,
            "paraphrased_all_elements": count == 4 and not literal}
