#!/usr/bin/env python3
"""Evaluate a saved model and emit span metrics broken down by label.

The input is Talmur source JSONL.  Use this for holdout and gold files only;
it has no training functionality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

from compile_jsonl_to_bio import compile_record, iter_jsonl, read_json
from finetune_token_classifier import build_labels, tokenize_and_align


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", help="Evaluate only this ISO language code")
    parser.add_argument("--split", help="Evaluate only records with this split value")
    parser.add_argument("--labels", help="Comma-separated base labels to score; other predictions become O")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    scored_labels = {item.strip() for item in args.labels.split(",") if item.strip()} if args.labels else None

    taxonomy = read_json(args.taxonomy)
    labels, label2id = build_labels(taxonomy)
    id2label = {index: label for index, label in enumerate(labels)}
    allowed = set(taxonomy["labels"])
    records = []
    for _, source in iter_jsonl(args.jsonl):
        if args.language and source.get("language") != args.language:
            continue
        if args.split and source.get("split") != args.split:
            continue
        records.append(compile_record(source, allowed))
    if not records:
        raise SystemExit("no records selected")

    dataset = Dataset.from_list(records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    tokenized = dataset.map(
        lambda batch: tokenize_and_align(batch, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(args.output.parent / ".eval_tmp"), per_device_eval_batch_size=args.batch_size, report_to="none"),
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    prediction = trainer.predict(tokenized)
    predicted_ids = np.argmax(prediction.predictions, axis=2)
    gold, predicted = [], []
    for row_predicted, row_gold in zip(predicted_ids, prediction.label_ids):
        gold_row, predicted_row = [], []
        for label_id, label in zip(row_predicted, row_gold):
            if label == -100:
                continue
            gold_label, predicted_label = id2label[int(label)], id2label[int(label_id)]
            if scored_labels:
                if gold_label != "O" and gold_label[2:] not in scored_labels:
                    gold_label = "O"
                if predicted_label != "O" and predicted_label[2:] not in scored_labels:
                    predicted_label = "O"
            gold_row.append(gold_label)
            predicted_row.append(predicted_label)
        gold.append(gold_row)
        predicted.append(predicted_row)

    report = classification_report(gold, predicted, output_dict=True, zero_division=0)
    overall = report.pop("micro avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
    report.pop("macro avg", None)
    report.pop("weighted avg", None)
    result = {
        "records": len(records), "language": args.language or "all", "source": str(args.jsonl), "scored_labels": sorted(scored_labels) if scored_labels else "all",
        "overall": {"precision": overall["precision"], "recall": overall["recall"], "f1": overall["f1-score"], "support": overall["support"]},
        "per_label": {name.removeprefix("B-"): {"precision": value["precision"], "recall": value["recall"], "f1": value["f1-score"], "support": value["support"]} for name, value in report.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=json_default)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
