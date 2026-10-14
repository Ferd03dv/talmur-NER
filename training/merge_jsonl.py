#!/usr/bin/env python3
"""Merge JSONL files preserving order."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as out:
        for raw in args.inputs:
            path = Path(raw)
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    print(f"merged {count} records into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
