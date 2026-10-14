#!/usr/bin/env python3
"""Evaluate a saved token-classification model on Talmur source JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

from compile_jsonl_to_bio import compile_record, iter_jsonl, read_json
from finetune_token_classifier import build_labels, make_compute_metrics, tokenize_and_align


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    taxonomy = read_json(args.taxonomy)
    labels, label2id = build_labels(taxonomy)
    id2label = {i: label for i, label in enumerate(labels)}
    allowed = set(taxonomy["labels"])

    records = [compile_record(record, allowed) for _, record in iter_jsonl(args.jsonl)]
    dataset = Dataset.from_list(records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    tokenized = dataset.map(
        lambda batch: tokenize_and_align(batch, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output.parent / ".eval_tmp"),
        per_device_eval_batch_size=8,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics(id2label),
    )
    metrics = trainer.evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
