#!/usr/bin/env python3
"""Fail-fast checks to run before submitting an expensive GPU training job."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def require_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"cannot import {name}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="microsoft/Multilingual-MiniLM-L12-H384")
    parser.add_argument("--taxonomy", default="taxonomy/deepmount_italian_ner_xxl_v2.labels.json", type=Path)
    parser.add_argument("--compiled-jsonl", default="data/compiled/train.seed.all.compiled.jsonl", type=Path)
    parser.add_argument("--min-records", type=int, default=1000)
    args = parser.parse_args()

    for module in [
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        "seqeval",
        "google.protobuf",
        "sentencepiece",
    ]:
        require_import(module)

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments

    if not args.taxonomy.exists():
        raise FileNotFoundError(f"taxonomy not found: {args.taxonomy}")
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    labels = taxonomy.get("labels") or []
    if len(labels) != 58:
        raise RuntimeError(f"expected 58 taxonomy labels, found {len(labels)}")

    if not args.compiled_jsonl.exists():
        raise FileNotFoundError(f"compiled dataset not found: {args.compiled_jsonl}")
    rows = count_jsonl(args.compiled_jsonl)
    if rows < args.min_records:
        raise RuntimeError(f"compiled dataset too small: {rows} records, expected at least {args.min_records}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("tokenizer is not fast; offset mapping is required for token classification")
    sample = tokenizer("preflight Ferdinando Del Vecchio", return_offsets_mapping=True)
    if "input_ids" not in sample or "offset_mapping" not in sample:
        raise RuntimeError("tokenizer did not return input_ids + offset_mapping")

    config = AutoConfig.from_pretrained(args.model_name)
    if not getattr(config, "model_type", None):
        raise RuntimeError("model config missing model_type")
    _model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=1 + (2 * len(labels)),
        ignore_mismatched_sizes=True,
    )
    del _model

    # Do not instantiate full weights here; downloading config/tokenizer is enough
    # to catch dependency/tokenizer failures before waiting for a GPU allocation.
    print("preflight ok")
    print(f"python deps ok: torch={torch.__version__}, transformers={transformers.__version__}")
    print(f"model weights ok: {args.model_name}, model_type={config.model_type}")
    print(f"dataset={args.compiled_jsonl}, records={rows}")
    print(f"taxonomy labels={len(labels)}")
    print(f"trainer classes import ok: {Trainer.__name__}, {TrainingArguments.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
