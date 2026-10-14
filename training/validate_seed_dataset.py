#!/usr/bin/env python3
"""Validate the seed dataset without requiring Node.js on HPC nodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", default="taxonomy/deepmount_italian_ner_xxl_v2.labels.json", type=Path)
    parser.add_argument("--jsonl", default="data/seed/it.deepmount_coverage.seed.jsonl", type=Path)
    args = parser.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    allowed = set(taxonomy["labels"])
    covered = set()
    errors = []
    count = 0

    for line_no, record in iter_jsonl(args.jsonl):
        count += 1
        record_id = record.get("id", f"line {line_no}")
        text = record.get("text")
        if not isinstance(text, str) or not text:
            errors.append(f"{record_id}: missing text")
            continue
        if record.get("language") != "it":
            errors.append(f"{record_id}: expected language=it")
        if record.get("split") not in {"train", "validation", "test"}:
            errors.append(f"{record_id}: invalid split")
        entities = record.get("entities")
        if not isinstance(entities, list):
            errors.append(f"{record_id}: entities must be an array")
            continue
        for entity in entities:
            label = entity.get("label")
            surface = entity.get("surface")
            if label not in allowed:
                errors.append(f"{record_id}: unknown label {label!r}")
            else:
                covered.add(label)
            if not isinstance(surface, str) or surface not in text:
                errors.append(f"{record_id}: surface not found: {surface!r}")
            if entity.get("sensitivity") not in {"none", "low", "medium", "high", "critical"}:
                errors.append(f"{record_id}: invalid sensitivity for {label}")
            if entity.get("action") not in {"keep", "review", "mask"}:
                errors.append(f"{record_id}: invalid action for {label}")

    missing = sorted(allowed - covered)
    if missing:
        errors.append("missing label coverage: " + ", ".join(missing))

    if errors:
        raise SystemExit("\n".join(errors))

    print(f"seed dataset ok: {count} records, {len(covered)}/{len(allowed)} labels covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
