#!/usr/bin/env python3
"""Fine-tune a Hugging Face token-classification model for Talmur NER."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

try:
    import evaluate
except Exception:  # pragma: no cover - optional in smoke runs
    evaluate = None

from compile_jsonl_to_bio import compile_record, iter_jsonl, read_json


def build_labels(taxonomy: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    labels = ["O"]
    for base in taxonomy["labels"]:
        labels.append(f"B-{base}")
    for base in taxonomy["labels"]:
        labels.append(f"I-{base}")
    return labels, {label: i for i, label in enumerate(labels)}


def load_compiled_records(jsonl_path: Path, taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = set(taxonomy["labels"])
    records = []
    for line_no, record in iter_jsonl(jsonl_path):
        try:
            records.append(compile_record(record, allowed))
        except Exception as exc:
            raise ValueError(f"{jsonl_path}:{line_no}: {exc}") from exc
    return records


def load_precompiled_records(jsonl_path: Path, taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = set(taxonomy["labels"])
    records = []
    for line_no, record in iter_jsonl(jsonl_path):
        if not isinstance(record.get("entities"), list):
            raise ValueError(f"{jsonl_path}:{line_no}: missing entities")
        for entity in record["entities"]:
            if entity.get("label") not in allowed:
                raise ValueError(f"{jsonl_path}:{line_no}: unknown label {entity.get('label')!r}")
            if not isinstance(entity.get("start"), int) or not isinstance(entity.get("end"), int):
                raise ValueError(f"{jsonl_path}:{line_no}: precompiled entity missing start/end")
        records.append(record)
    return records


def entity_label_for_token(offset: tuple[int, int], entities: list[dict[str, Any]]) -> str:
    start, end = offset
    if start == end:
        return "IGNORED"
    for entity in entities:
        if start < entity["end"] and end > entity["start"]:
            prefix = "B" if start == entity["start"] else "I"
            return f"{prefix}-{entity['label']}"
    return "O"


def tokenize_and_align(batch, tokenizer, label2id, max_length: int):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    aligned_labels = []
    for offsets, entities in zip(tokenized["offset_mapping"], batch["entities"]):
        row = []
        for offset in offsets:
            label = entity_label_for_token(tuple(offset), entities)
            row.append(-100 if label == "IGNORED" else label2id[label])
        aligned_labels.append(row)
    tokenized.pop("offset_mapping")
    tokenized["labels"] = aligned_labels
    return tokenized


def split_dataset(records: list[dict[str, Any]]) -> DatasetDict:
    by_split = {"train": [], "validation": [], "test": []}
    for record in records:
        by_split.setdefault(record["split"], []).append(record)
    if not by_split["validation"]:
        by_split["validation"] = by_split["train"][:]
    data = {name: Dataset.from_list(rows) for name, rows in by_split.items() if rows}
    return DatasetDict(data)


def make_compute_metrics(id2label: dict[int, str]):
    seqeval = evaluate.load("seqeval") if evaluate is not None else None

    def compute_metrics(pred):
        predictions = np.argmax(pred.predictions, axis=2)
        true_predictions = []
        true_labels = []
        for pred_row, label_row in zip(predictions, pred.label_ids):
            out_pred = []
            out_label = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                out_pred.append(id2label[int(pred_id)])
                out_label.append(id2label[int(label_id)])
            true_predictions.append(out_pred)
            true_labels.append(out_label)
        if seqeval is None:
            total = sum(len(row) for row in true_labels)
            correct = sum(
                p == y
                for pred_row, label_row in zip(true_predictions, true_labels)
                for p, y in zip(pred_row, label_row)
            )
            return {"token_accuracy": correct / total if total else 0.0}
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results.get("overall_precision", 0.0),
            "recall": results.get("overall_recall", 0.0),
            "f1": results.get("overall_f1", 0.0),
            "accuracy": results.get("overall_accuracy", 0.0),
        }

    return compute_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--taxonomy", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-jsonl", type=Path, help="Source JSONL with surfaces; compiled internally.")
    source.add_argument("--compiled-jsonl", type=Path, help="Precompiled JSONL with entity start/end offsets.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    taxonomy = read_json(args.taxonomy)
    labels, label2id = build_labels(taxonomy)
    id2label = {i: label for i, label in enumerate(labels)}

    if args.compiled_jsonl:
        records = load_precompiled_records(args.compiled_jsonl, taxonomy)
    else:
        records = load_compiled_records(args.train_jsonl, taxonomy)
    dataset = split_dataset(records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("fast tokenizer is required because token/entity alignment needs offset_mapping")
    tokenized = dataset.map(
        lambda batch: tokenize_and_align(batch, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label={str(i): label for i, label in id2label.items()},
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics(id2label),
    )

    trainer.train()
    metrics = trainer.evaluate() if "validation" in tokenized else {}
    trainer.save_model(str(args.output_dir / "model"))
    tokenizer.save_pretrained(str(args.output_dir / "model"))

    metadata = {
        "model_name": args.model_name,
        "labels": labels,
        "train_records": len(dataset["train"]),
        "validation_records": len(dataset.get("validation", [])),
        "metrics": metrics,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
