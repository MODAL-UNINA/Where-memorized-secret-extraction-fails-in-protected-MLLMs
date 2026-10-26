"""Trusted E10 output filters; the attacker-facing runner never loads this store.

Both filters operate against the full private store of one target split.  The
literal arm redacts an exact sentence.  The element arm blocks a response when
it contains all four fields of any one secret, in any order or wording.
"""
from __future__ import annotations

import re

from e10_secret import SCHEME, parse_secret, present_fields


class FilterStore:
    def __init__(self, keys: list[dict]):
        if not keys or any(row.get("secret_scheme") != SCHEME for row in keys):
            raise ValueError("E10 filter requires private natural-language keys")
        if len({row["secret"] for row in keys}) != len(keys):
            raise ValueError("duplicate E10 secret in target store")
        self.secrets = []
        self.elements = []
        for row in keys:
            elements = parse_secret(row["secret"])
            if elements != row.get("elements"):
                raise ValueError("E10 key fields differ from sentence")
            self.secrets.append(row["secret"])
            self.elements.append(elements)
        self.literal_pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(secret) for secret in
                                    sorted(self.secrets, key=len, reverse=True)) + r")(?!\w)", re.I)

    def apply(self, response: str, filter_name: str) -> str:
        if filter_name == "none":
            return response
        if filter_name == "literal":
            return self.literal_pattern.sub("[REDACTED]", response)
        if filter_name == "elements":
            return "[BLOCKED]" if any(all(present_fields(response, values).values())
                                      for values in self.elements) else response
        raise ValueError(f"unknown E10 filter: {filter_name}")
