#!/usr/bin/env python3
"""Import DeepMount00/pii-masking-ita Parquet splits into Talmur compiled JSONL.

The upstream dataset stores character spans in a Python-literal ``span_labels``
column.  This importer reads those spans directly, maps only semantically
compatible labels to the fixed Talmur taxonomy, and keeps the upstream test
split out of training.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


# Deliberately conservative: labels with no clean equivalent are omitted rather
# than teaching the classifier an incorrect semantic association.
LABEL_MAP = {
    "ACCOUNTNUMBER": "NUMERO_CONTO",
    "AGE": "ETA",
    "AMOUNT": "IMPORTO",
    "BIC": "BIC",
    "BUILDINGNUMBER": "INDIRIZZO",
    "CITY": "LUOGO",
    "COMPANYNAME": "RAGIONE_SOCIALE",
    "COUNTY": "LUOGO",
    "CREDITCARDCVV": "CVV",
    "CREDITCARDISSUER": "BRAND",
    "CREDITCARDNUMBER": "NUMERO_CARTA",
    "CURRENCY": "VALUTA",
    "CURRENCYCODE": "VALUTA",
    "CURRENCYNAME": "VALUTA",
    "CURRENCYSYMBOL": "VALUTA",
    "DATE": "DATA",
    "DOB": "DATA_NASCITA",
    "EMAIL": "EMAIL",
    "FIRSTNAME": "NOME",
    "IBAN": "IBAN",
    "IP": "IP",
    "IPV4": "IP",
    "IPV6": "IPV6_1",
    "JOBTITLE": "PROFESSIONE",
    "LASTNAME": "COGNOME",
    "MAC": "MAC",
    "PASSWORD": "PASSWORD",
    "PHONEIMEI": "IMEI",
    "PHONENUMBER": "NUMERO_TELEFONO",
    "PIN": "PIN",
    "SECONDARYADDRESS": "INDIRIZZO",
    "SSN": "NUMERO_DOCUMENTO",
    "STATE": "STATO",
    "STREET": "INDIRIZZO",
    "TIME": "ORARIO",
    "URL": "URL",
    "USERAGENT": "USER_AGENT",
    "VEHICLEVRM": "TARGA_VEICOLO",
    "ZIPCODE": "CODICE_POSTALE",
}

INDEXED_LABEL_RE = re.compile(r"^(?P<label>.+?)_\d+$")


def read_taxonomy(path: Path) -> set[str]:
    return set(json.loads(path.read_text(encoding="utf-8"))["labels"])


def base_label(indexed_label: str) -> str:
    match = INDEXED_LABEL_RE.match(indexed_label)
    return match.group("label") if match else indexed_label


def split_for_train_row(row_number: int, seed: int, validation_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{row_number}".encode("ascii")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if value < validation_ratio else "train"


def convert_row(
    row: dict[str, Any],
    row_number: int,
    split: str,
    allowed_labels: set[str],
) -> dict[str, Any]:
    text = row["unmasked_text"]
    spans = ast.literal_eval(row["span_labels"])
    entities: list[dict[str, Any]] = []
    for start, end, upstream_label in spans:
        if upstream_label == "O":
            continue
        target_label = LABEL_MAP.get(base_label(upstream_label))
        if target_label is None:
            continue
        if target_label not in allowed_labels:
            raise ValueError(f"mapped label missing from taxonomy: {target_label}")
        surface = text[start:end]
        if not surface:
            raise ValueError(f"empty span at row {row_number}: {upstream_label}")
        entities.append(
            {
                "label": target_label,
                "start": int(start),
                "end": int(end),
                "surface": surface,
                "upstream_label": base_label(upstream_label),
            }
        )
    entities.sort(key=lambda entity: (entity["start"], entity["end"]))
    return {
        "id": f"deepmount-ita-{split}-{row_number:06d}",
        "language": "it",
        "split": split,
        "text": text,
        "entities": entities,
        "source": {"dataset": "DeepMount00/pii-masking-ita", "row": row_number},
    }


def import_split(
    input_path: Path,
    output_file,
    default_split: str,
    allowed_labels: set[str],
    seed: int,
    validation_ratio: float,
    label_counts: Counter[str],
) -> tuple[int, Counter[str]]:
    rows = pq.read_table(input_path, columns=["unmasked_text", "span_labels"]).to_pylist()
    records = 0
    skipped = Counter()
    for row_number, row in enumerate(rows):
        split = (
            split_for_train_row(row_number, seed, validation_ratio)
            if default_split == "train"
            else default_split
        )
        record = convert_row(row, row_number, split, allowed_labels)
        for entity in record["entities"]:
            label_counts[entity["label"]] += 1
        output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        records += 1
    return records, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--test-parquet", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")
    allowed_labels = read_taxonomy(args.taxonomy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.output.open("w", encoding="utf-8") as output_file:
        train_records, _ = import_split(
            args.train_parquet,
            output_file,
            "train",
            allowed_labels,
            args.seed,
            args.validation_ratio,
            counts,
        )
        test_records, _ = import_split(
            args.test_parquet,
            output_file,
            "test",
            allowed_labels,
            args.seed,
            args.validation_ratio,
            counts,
        )
    print(f"wrote {train_records + test_records} records to {args.output}")
    print(f"source splits: train={train_records}, test={test_records}")
    print("mapped entity counts: " + ", ".join(f"{label}={counts[label]}" for label in sorted(counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
