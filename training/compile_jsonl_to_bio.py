#!/usr/bin/env python3
"""Compile Talmur source JSONL records to character offsets and simple BIO tokens.

The source dataset stores entity surfaces instead of offsets to keep manual seed
records readable. This script resolves surfaces to character offsets and emits a
compiled JSONL suitable for inspection and downstream tokenization.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\S+")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def find_occurrence(text: str, surface: str, occurrence: int = 1) -> tuple[int, int]:
    if occurrence < 1:
        raise ValueError("occurrence must be >= 1")
    start = -1
    cursor = 0
    for _ in range(occurrence):
        start = text.find(surface, cursor)
        if start == -1:
            raise ValueError(f"surface not found: {surface!r}")
        cursor = start + len(surface)
    return start, start + len(surface)


def overlaps(start: int, end: int, occupied: list[tuple[int, int, str]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end, _ in occupied)


def find_non_overlapping_occurrence(
    text: str,
    surface: str,
    occupied: list[tuple[int, int, str]],
    occurrence: int = 1,
) -> tuple[int, int]:
    """Find requested occurrence, or the next occurrence that does not overlap.

    Source records can specify `occurrence` explicitly. When omitted, this still
    avoids common synthetic-data ambiguities such as city names inside street
    names (`Via Roma 12, Roma`) or account numbers inside IBANs.
    """
    cursor = 0
    seen = 0
    while True:
        start = text.find(surface, cursor)
        if start == -1:
            raise ValueError(f"surface not found without overlap: {surface!r}")
        end = start + len(surface)
        seen += 1
        cursor = end
        if seen < occurrence:
            continue
        if not overlaps(start, end, occupied):
            return start, end


def compile_record(record: dict[str, Any], allowed_labels: set[str]) -> dict[str, Any]:
    text = record.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{record.get('id', '<missing id>')}: text must be a non-empty string")

    compiled_entities = []
    occupied: list[tuple[int, int, str]] = []
    for entity in record.get("entities", []):
        label = entity.get("label")
        if label not in allowed_labels:
            raise ValueError(f"{record.get('id')}: unknown label {label!r}")
        surface = entity.get("surface")
        if not isinstance(surface, str) or not surface:
            raise ValueError(f"{record.get('id')}: entity surface must be non-empty")
        occurrence = int(entity.get("occurrence", 1))
        start, end = find_non_overlapping_occurrence(text, surface, occupied, occurrence)
        occupied.append((start, end, label))
        compiled_entities.append({**entity, "start": start, "end": end})

    compiled_entities.sort(key=lambda e: (e["start"], e["end"]))
    tokens = []
    for match in TOKEN_RE.finditer(text):
        start, end = match.span()
        token_label = "O"
        for entity in compiled_entities:
            if start < entity["end"] and end > entity["start"]:
                prefix = "B" if start == entity["start"] else "I"
                token_label = f"{prefix}-{entity['label']}"
                break
        tokens.append({"text": match.group(0), "start": start, "end": end, "label": token_label})

    return {
        "id": record["id"],
        "language": record["language"],
        "split": record["split"],
        "source": record.get("source", {}),
        "text": text,
        "entities": compiled_entities,
        "tokens": tokens,
        "record_tags": record.get("record_tags", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    taxonomy = read_json(args.taxonomy)
    allowed = set(taxonomy["labels"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    labels = set()
    with args.output.open("w", encoding="utf-8") as out:
        for line_no, record in iter_jsonl(args.input):
            try:
                compiled = compile_record(record, allowed)
            except Exception as exc:
                raise ValueError(f"{args.input}:{line_no}: {exc}") from exc
            for entity in compiled["entities"]:
                labels.add(entity["label"])
            out.write(json.dumps(compiled, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1

    missing = sorted(allowed - labels)
    print(f"compiled {count} records to {args.output}")
    print(f"covered labels: {len(labels)}/{len(allowed)}")
    if missing:
        print("missing labels: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
