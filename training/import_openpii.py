#!/usr/bin/env python3
"""Stream a language-filtered, conservatively mapped OpenPII snapshot.

OpenPII is published as large, monolithic JSONL files.  This tool deliberately
does not store those originals: it streams one split, keeps only requested
languages, and maps only labels whose meaning is safe for the Talmur taxonomy.

Do not map TAXNUM globally: its meaning depends on the country and it cannot
reliably be split between CODICE_FISCALE and P_IVA without locale-aware rules.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


DATASET = "ai4privacy/pii-masking-openpii-1.5m"
BASE_URL = "https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m/resolve/main/data/{split}.jsonl?download=true"

# These are deliberately one-to-one or low-risk many-to-one mappings.  Labels
# such as TAXNUM, TITLE, SEX and GENDER are not emitted as Talmur NER entities.
LABEL_MAP = {
    "DATE": "DATA",
    "AGE": "ETA",
    "GIVENNAME": "NOME",
    "SURNAME": "COGNOME",
    "STREET": "INDIRIZZO",
    "EMAIL": "EMAIL",
    "ZIPCODE": "CODICE_POSTALE",
    "CITY": "LUOGO",
    "SOCIALNUM": "NUMERO_DOCUMENTO",
    "IDCARDNUM": "NUMERO_DOCUMENTO",
    "PASSPORTNUM": "NUMERO_DOCUMENTO",
    "CREDITCARDNUMBER": "NUMERO_CARTA",
    "TELEPHONENUM": "NUMERO_TELEFONO",
    "DRIVERLICENSENUM": "N_LICENZA",
}

CANONICAL = {
    "DATA": "DATE", "ETA": "AGE", "NOME": "PERSON_FIRSTNAME",
    "COGNOME": "PERSON_LASTNAME", "INDIRIZZO": "ADDRESS",
    "EMAIL": "EMAIL_ADDRESS", "CODICE_POSTALE": "POSTAL_CODE",
    "LUOGO": "LOCATION", "NUMERO_DOCUMENTO": "DOCUMENT_NUMBER",
    "NUMERO_CARTA": "CREDIT_CARD", "NUMERO_TELEFONO": "PHONE_NUMBER",
    "N_LICENZA": "LICENSE_NUMBER",
}


def occurrence_in_text(text: str, surface: str, start: int) -> int:
    """Return the one-based occurrence whose start is ``start``."""
    cursor = 0
    occurrence = 0
    while True:
        found = text.find(surface, cursor)
        if found < 0 or found > start:
            return max(1, occurrence)
        occurrence += 1
        if found == start:
            return occurrence
        cursor = found + len(surface)


def is_adjacent_address_part(left: dict[str, Any], right: dict[str, Any], text: str) -> bool:
    between = text[int(left["end"]):int(right["start"])]
    return bool(re.fullmatch(r"[\s,.-]*", between))


def map_entities(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Map OpenPII spans, merging adjacent house-number/street spans."""
    text = record["source_text"]
    masks = sorted(record.get("privacy_mask", []), key=lambda item: (item["start"], item["end"]))
    entities: list[dict[str, Any]] = []
    consumed: set[int] = set()

    for index, mask in enumerate(masks):
        if index in consumed:
            continue
        source_label = mask.get("label")
        if source_label == "BUILDINGNUM":
            # Merge a number with the following street if they are contiguous.
            next_index = index + 1
            if next_index < len(masks) and masks[next_index].get("label") == "STREET" and is_adjacent_address_part(mask, masks[next_index], text):
                street = masks[next_index]
                start, end = int(mask["start"]), int(street["end"])
                surface = text[start:end]
                entities.append(make_entity(text, surface, start, "INDIRIZZO", "STREET+BUILDINGNUM"))
                consumed.add(next_index)
            continue
        if source_label == "STREET":
            # Also support the less common reversed order: street then number.
            next_index = index + 1
            if next_index < len(masks) and masks[next_index].get("label") == "BUILDINGNUM" and is_adjacent_address_part(mask, masks[next_index], text):
                number = masks[next_index]
                start, end = int(mask["start"]), int(number["end"])
                surface = text[start:end]
                entities.append(make_entity(text, surface, start, "INDIRIZZO", "STREET+BUILDINGNUM"))
                consumed.add(next_index)
            else:
                entities.append(make_entity(text, str(mask["value"]), int(mask["start"]), "INDIRIZZO", "STREET"))
            continue
        target = LABEL_MAP.get(source_label)
        if target:
            entities.append(make_entity(text, str(mask["value"]), int(mask["start"]), target, str(source_label)))

    # The Talmur compiler rejects overlaps.  OpenPII normally has none, but the
    # merge above can create one in malformed input, so retain the first span.
    entities.sort(key=lambda item: (item["_start"], -item["_end"]))
    non_overlapping: list[dict[str, Any]] = []
    end = -1
    for entity in entities:
        if entity["_start"] >= end:
            end = entity["_end"]
            entity.pop("_start")
            entity.pop("_end")
            non_overlapping.append(entity)
    return non_overlapping


def make_entity(text: str, surface: str, start: int, label: str, source_label: str) -> dict[str, Any]:
    end = start + len(surface)
    if text[start:end] != surface:
        raise ValueError(f"invalid OpenPII span for {source_label}: {surface!r}")
    return {
        "surface": surface,
        "label": label,
        "canonical_entity": CANONICAL[label],
        "occurrence": occurrence_in_text(text, surface, start),
        "sensitivity": "critical" if label in {"NUMERO_CARTA", "NUMERO_DOCUMENTO"} else "high",
        "action": "mask",
        "context_tags": ["openpii", source_label.lower()],
        "notes": f"mapped from OpenPII {source_label}",
        "_start": start,
        "_end": end,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split", choices=["train", "validation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", default="it,en,fr,de,es,pt")
    parser.add_argument("--max-per-language", type=int, default=25_000)
    parser.add_argument("--resume", action="store_true", help="Append to an interrupted output without duplicating record ids.")
    args = parser.parse_args()
    if args.max_per_language < 1:
        raise SystemExit("--max-per-language must be positive")

    languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
    wanted = set(languages)
    counts = collections.Counter()
    existing_ids: set[str] = set()
    skipped_unmapped = 0
    scanned = 0
    url = BASE_URL.format(split=args.source_split)
    request = urllib.request.Request(url, headers={"User-Agent": "talmur-ner-openpii-import/1.0"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                saved = json.loads(line)
                language = saved.get("language")
                if language in wanted:
                    counts[language] += 1
                    existing_ids.add(str(saved.get("id")))
        print("resuming with existing records: " + ", ".join(f"{lang}={counts[lang]}" for lang in languages), flush=True)

    mode = "a" if args.resume else "w"
    with urllib.request.urlopen(request, timeout=120) as response, args.output.open(mode, encoding="utf-8") as out:
        for raw_line in response:
            scanned += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON received after {scanned} rows") from exc
            language = record.get("language")
            if language not in wanted or counts[language] >= args.max_per_language:
                continue
            entities = map_entities(record)
            if not entities:
                skipped_unmapped += 1
                continue
            uid = record.get("uid", scanned)
            record_id = f"openpii-{args.source_split}-{uid}"
            if record_id in existing_ids:
                continue
            result = {
                "id": record_id,
                "language": language,
                "split": args.source_split,
                "source": {
                    "kind": "public_dataset",
                    "template_id": "openpii_1_5m_conservative_map_v1",
                    "document_type": "openpii_synthetic_document",
                    "notes": f"{DATASET}; source split={args.source_split}; CC-BY-4.0",
                },
                "text": record["source_text"],
                "entities": entities,
                "record_tags": ["openpii_1_5m", "conservative_mapping", language],
            }
            out.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[language] += 1
            if all(counts[lang] >= args.max_per_language for lang in languages):
                break

    print(f"wrote {sum(counts.values())} records to {args.output}")
    print("language counts: " + ", ".join(f"{lang}={counts[lang]}" for lang in languages))
    print(f"scanned source rows: {scanned}; skipped rows without conservative mappings: {skipped_unmapped}")
    if any(counts[lang] < args.max_per_language for lang in languages):
        print("warning: end of source reached before requested count for at least one language", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
