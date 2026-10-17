#!/usr/bin/env python3
"""Create a deterministic, language-balanced OpenPII snapshot from an import."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def bucket(record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(record_id.encode("utf-8")).digest()[:8], "big")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", default="it,en,fr,de,es,pt")
    parser.add_argument("--per-language", type=int, default=10_000)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.per_language < 1 or not 0 < args.validation_ratio < 1:
        raise SystemExit("per-language must be positive and validation ratio must be between 0 and 1")

    languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
    selected = {language: [] for language in languages}
    with args.input.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            language = record.get("language")
            if language in selected:
                selected[language].append(record)

    missing = {language: args.per_language - len(rows) for language, rows in selected.items() if len(rows) < args.per_language}
    if missing:
        raise SystemExit("insufficient records: " + ", ".join(f"{lang} needs {count} more" for lang, count in missing.items()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0}
    with args.output.open("w", encoding="utf-8") as out:
        for language in languages:
            rows = sorted(selected[language], key=lambda row: bucket(row["id"]))[:args.per_language]
            validation_count = round(args.per_language * args.validation_ratio)
            for index, row in enumerate(rows):
                row["split"] = "validation" if index < validation_count else "train"
                row["record_tags"] = [*row.get("record_tags", []), "balanced_snapshot_v1"]
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[row["split"]] += 1
    print(f"wrote {sum(counts.values())} records to {args.output}")
    print("split counts: " + ", ".join(f"{name}={value}" for name, value in counts.items()))
    print("per language: " + ", ".join(f"{language}={args.per_language}" for language in languages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
