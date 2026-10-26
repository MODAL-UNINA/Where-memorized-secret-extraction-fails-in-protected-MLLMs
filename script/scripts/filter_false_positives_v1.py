#!/usr/bin/env python3
"""False positives of the frozen output filters on benign text that no target produced.

The utility ledgers already show that the filters never change a benign model
response (counterfactual analysis), but those responses are short arithmetic
and sorting answers. This test applies the same filters to two external
corpora, split into windows of about one response:

* natural: the 50,000 Wikipedia and C4-news passages of
  ``Dataset/train_integrated_text.jsonl`` (canary records excluded);
* technical: hex-dense text from the analysis workstation (CPython standard
  library source, C headers under /usr/include, Debian package changelogs),
  where identifiers resembling ``ID-`` plus eight hex digits are most likely.

Filters: the primary format regex (``protected_interface.filter_output``), the
exact and separator-normalized known-value filters of
``filter_counterfactual_v1`` with the union of all eval key stores (Qwen
101-105, which LLaVA v3 101-103 share), and the E10 literal and four-field
filters with the union of the E10 eval stores (seeds 106, 107). A window is a
false positive when a filter changes it. Matched benign strings of the format
filter are recorded; nothing derived from a private key is written.

CPU only; reads private keys to build the value filters.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import sys
import sysconfig
from pathlib import Path

import defense_decomposition_v1 as dd
from e10_filters import FilterStore
from filter_counterfactual_v1 import SEEDS, Store, store_for
from protected_interface import FILTER, filter_output

ROOT = dd.ROOT
PRIVATE = ROOT / "private"
OUTPUT = PRIVATE / "filter_false_positives_v1.json"
NATURAL = ROOT / "Dataset/train_integrated_text.jsonl"
E10_STORES = (PRIVATE / "e10/target_seed_106/partition_seed_1106/eval_all_keys.jsonl",
              PRIVATE / "e10/target_seed_107/partition_seed_1107/eval_all_keys.jsonl")
WINDOW = 2000  # characters; longer than any benign utility response
FILTERS = ("format", "exact_value", "normalized_value", "e10_literal", "e10_elements")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def windows(text: str) -> list[str]:
    return [text[i:i + WINDOW] for i in range(0, len(text), WINDOW)] or [""]


def natural_corpus() -> tuple[list[str], dict]:
    docs = [f"{row['prefix']} {row['target']}" for row in read_jsonl(NATURAL) if row["source"] != "canary_text"]
    return docs, {"path": str(NATURAL.relative_to(ROOT)), "sha256": dd.sha(NATURAL), "documents": len(docs)}


def technical_files() -> list[Path]:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(stdlib.rglob("*.py")) + sorted(Path("/usr/include").rglob("*.h"))
    files += sorted(Path("/usr/share/doc").glob("*/changelog*.gz"))
    return [f for f in files if f.is_file() and not f.is_symlink()]


def technical_corpus() -> tuple[list[str], dict]:
    docs, digest = [], hashlib.sha256()
    for path in technical_files():
        raw = path.read_bytes()
        text = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8", "replace")
        docs.append(text)
        digest.update(str(path).encode() + b"\0" + hashlib.sha256(raw).digest())
    return docs, {"sources": ["CPython stdlib *.py", "/usr/include **/*.h", "/usr/share/doc/*/changelog*.gz"],
                  "python": sys.version.split()[0], "platform": platform.platform(),
                  "files": len(docs), "manifest_sha256": digest.hexdigest()}


def build_filters():
    qwen = [store_for(seed, partition, "eval")[0] for seed, partition in SEEDS]
    values = Store({value for store in qwen for value in store.values})
    e10 = FilterStore([row for path in E10_STORES for row in read_jsonl(path)])
    return {"format": filter_output,
            "exact_value": lambda text: values.apply("exact_value", text),
            "normalized_value": lambda text: values.apply("normalized_value", text),
            "e10_literal": lambda text: e10.apply(text, "literal"),
            "e10_elements": lambda text: e10.apply(text, "elements")}, {
        "value_store_size": len(values.values), "e10_store_size": len(e10.secrets),
        "e10_store_sha256": [dd.sha(path) for path in E10_STORES]}


def measure(name: str, docs: list[str], filters: dict) -> dict:
    hits = {f: 0 for f in FILTERS}
    doc_hits = {f: 0 for f in FILTERS}
    matches: dict[str, int] = {}
    total = chars = 0
    for doc in docs:
        changed_doc = set()
        for window in windows(doc):
            total += 1
            chars += len(window)
            for f in FILTERS:
                if filters[f](window) != window:
                    hits[f] += 1
                    changed_doc.add(f)
            for match in FILTER.finditer(window):
                matches[match.group(0)] = matches.get(match.group(0), 0) + 1
        for f in changed_doc:
            doc_hits[f] += 1
    return {"corpus": name, "documents": len(docs), "windows": total, "characters": chars,
            "window_chars": WINDOW,
            "filters": {f: {"windows_changed": hits[f], "rate": hits[f] / total,
                            "wilson95": dd.wilson(hits[f], total), "documents_changed": doc_hits[f]}
                        for f in FILTERS},
            "format_matches": dict(sorted(matches.items(), key=lambda kv: -kv[1]))}


def main() -> None:
    filters, store_info = build_filters()
    natural, natural_info = natural_corpus()
    technical, technical_info = technical_corpus()
    result = {"test": "benign-text false positives of frozen output filters",
              "filter_sha256": {"protected_interface": dd.sha(Path(__file__).with_name("protected_interface.py")),
                                "filter_counterfactual_v1": dd.sha(Path(__file__).with_name("filter_counterfactual_v1.py")),
                                "e10_filters": dd.sha(Path(__file__).with_name("e10_filters.py"))},
              "stores": store_info,
              "corpora": {"natural": {**natural_info, **measure("natural", natural, filters)},
                          "technical": {**technical_info, **measure("technical", technical, filters)}}}
    with OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    OUTPUT.chmod(0o600)
    for name, corpus in result["corpora"].items():
        print(f"{name}: {corpus['documents']} docs, {corpus['windows']} windows, {corpus['characters']:,} chars")
        for f, cell in corpus["filters"].items():
            print(f"  {f:17s} {cell['windows_changed']:6d} windows ({cell['rate']:.2e}) "
                  f"upper95={cell['wilson95'][1]:.2e} docs={cell['documents_changed']}")
        print("  format matches:", list(corpus["format_matches"].items())[:10])


if __name__ == "__main__":
    main()
