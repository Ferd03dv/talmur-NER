#!/usr/bin/env python3
"""Measure end-to-end CPU inference latency for a saved Talmur NER model."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


DEFAULT_TEXT = "Buongiorno, sono Anna Verdi. Scrivete a anna.verdi@example.test per la pratica 8472."


def percentile(values: list[float], value: float) -> float:
    """Nearest-rank percentile without an extra dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * value)))
    return ordered[index]


def read_jsonl_texts(path: Path, max_samples: int) -> list[str]:
    texts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        text = record.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    if not texts:
        raise ValueError(f"no usable texts in {path}")
    return texts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--jsonl", type=Path, help="JSONL whose text fields are benchmark inputs.")
    source.add_argument("--text", action="append", help="Text to benchmark; may be passed more than once.")
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3, help="Measurements per input text.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 keeps PyTorch's default.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1 or args.max_samples < 1:
        raise ValueError("warmup must be >= 0; iterations and max-samples must be >= 1")
    if args.threads < 0:
        raise ValueError("threads must be >= 0")
    if args.threads:
        torch.set_num_threads(args.threads)

    texts = read_jsonl_texts(args.jsonl, args.max_samples) if args.jsonl else (args.text or [DEFAULT_TEXT])
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    def infer(text: str) -> int:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        with torch.inference_mode():
            model(**encoded)
        return int(encoded["input_ids"].shape[1])

    for step in range(args.warmup):
        infer(texts[step % len(texts)])

    latencies_ms: list[float] = []
    token_counts: list[int] = []
    started = time.perf_counter()
    for _ in range(args.iterations):
        for text in texts:
            before = time.perf_counter()
            token_counts.append(infer(text))
            latencies_ms.append((time.perf_counter() - before) * 1000)
    elapsed = time.perf_counter() - started

    result = {
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "model_dir": str(args.model_dir),
        "inputs": len(texts),
        "measurements": len(latencies_ms),
        "warmup_runs": args.warmup,
        "mean_tokens": statistics.mean(token_counts),
        "latency_ms": {
            "mean": statistics.mean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
        },
        "documents_per_second": len(latencies_ms) / elapsed if elapsed else 0.0,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
