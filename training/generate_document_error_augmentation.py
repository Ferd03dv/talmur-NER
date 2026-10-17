#!/usr/bin/env python3
"""Generate Italian document-style augmentation from observed error families.

The records deliberately contain fictitious names, dates and organisations. They
model CV, bill and business-document layouts without copying personal data from
the source documents. Each family has equally sized O-only hard negatives.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CANONICAL = {"COGNOME": "PERSON_LASTNAME", "DATA": "DATE", "LUOGO": "LOCATION", "PROFESSIONE": "PROFESSION"}
DATES = ("FEB 2026", "APR 2026", "Febbraio 2026", "Aprile 2026", "02/26", "04/26", "01/2026", "SET 2025", "Ottobre 2024", "NOV. 2023")
SURNAMES = ("Rossi", "Bianchi", "Esposito", "Romano", "Ferrari", "De Luca", "Del Vecchio")
PROFESSIONS = ("Avv.", "Avvocato", "Ing.", "Ingegnere", "Dottore", "Dottoressa")
CITIES = ("Milano", "Torino", "Bologna", "Firenze", "Napoli", "Bari", "Verona")

DATE_TEMPLATES = ("Periodo di fatturazione: {date}.", "Business plan aggiornato a {date}.", "Contratto con decorrenza {date}.", "Documento emesso nel mese di {date}.")
PROFILE_TEMPLATES = ("Curriculum vitae di {profession} {surname}.", "Il candidato {profession} {surname} ricopre il ruolo indicato nel profilo.", "Intestatario della pratica: {profession} {surname}.", "Firma del professionista: {profession} {surname}.")
LOCATION_TEMPLATES = ("Sede operativa: {city}.", "Comune di riferimento: {city}.", "Indirizzo di recapito presso la filiale di {city}.", "La riunione con il cliente si terrà a {city}.")

# Headings and boilerplate observed in the supplied document types. None is a location.
LOCATION_NEGATIVES = ("Business Plan: visione strategica e piano operativo per il mercato libero.", "Il servizio Bolletta Web usa un riferimento interno per la verifica.", "Energia elettrica: riepilogo consumi, condizioni economiche e assistenza clienti.", "Pagina di sintesi del progetto, con obiettivi e piano di sviluppo.", "Il centro di assistenza gestisce la richiesta tramite il servizio clienti.")
DATE_NEGATIVES = ("La sigla FEB è usata soltanto come codice interno del reparto.", "Il campo 02/26 è un riferimento di archivio, non una data del documento.", "Il piano usa APR come abbreviazione del progetto, senza indicare una data.", "La voce Ottobre identifica il nome di una procedura interna.")
PROFILE_NEGATIVES = ("Il profilo descrive competenze tecniche e responsabilità del team.", "La pratica è stata esaminata dall'ufficio competente.", "Il documento contiene un riepilogo delle attività svolte.", "La firma digitale è stata verificata correttamente dal sistema.")


def split_for(index: int, validation_ratio: float) -> str:
    return "validation" if index % round(1 / validation_ratio) == 0 else "train"


def entity(surface: str, label: str) -> dict:
    return {"surface": surface, "label": label, "canonical_entity": CANONICAL[label], "sensitivity": "high", "action": "mask", "context_tags": ["document_error_augmentation", label.lower()]}


def record(record_id: str, split: str, text: str, entities: list[dict], family: str, polarity: str) -> dict:
    return {
        "id": record_id, "language": "it", "split": split,
        "source": {"kind": "synthetic_document_error_augmentation", "template_id": family, "document_type": "cv_billing_business_fragment", "notes": "Fictitious data derived from observed error patterns; no source-document PII."},
        "text": text, "entities": entities,
        "record_tags": ["document_error_augmentation_v1", family, polarity, "it"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--examples-per-family", type=int, default=600)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if args.examples_per_family < 1 or not 0 < args.validation_ratio < 1:
        raise SystemExit("examples-per-family must be positive and validation-ratio must be between 0 and 1")

    rng = random.Random(args.seed)
    records: list[dict] = []
    for index in range(1, args.examples_per_family + 1):
        split = split_for(index, args.validation_ratio)
        date = rng.choice(DATES)
        records += [
            record(f"doc-error-date-positive-{index:05d}", split, rng.choice(DATE_TEMPLATES).format(date=date), [entity(date, "DATA")], "document_date", "positive"),
            record(f"doc-error-date-negative-{index:05d}", split, f"{rng.choice(DATE_NEGATIVES)} Riferimento {index:05d}.", [], "document_date", "hard_negative"),
        ]
        profession, surname = rng.choice(PROFESSIONS), rng.choice(SURNAMES)
        records += [
            record(f"doc-error-profile-positive-{index:05d}", split, rng.choice(PROFILE_TEMPLATES).format(profession=profession, surname=surname), [entity(profession, "PROFESSIONE"), entity(surname, "COGNOME")], "profession_surname", "positive"),
            record(f"doc-error-profile-negative-{index:05d}", split, f"{rng.choice(PROFILE_NEGATIVES)} Riferimento {index:05d}.", [], "profession_surname", "hard_negative"),
        ]
        city = rng.choice(CITIES)
        records += [
            record(f"doc-error-location-positive-{index:05d}", split, rng.choice(LOCATION_TEMPLATES).format(city=city), [entity(city, "LUOGO")], "location_precision", "positive"),
            record(f"doc-error-location-negative-{index:05d}", split, f"{rng.choice(LOCATION_NEGATIVES)} Riferimento {index:05d}.", [], "location_precision", "hard_negative"),
        ]

    rng.shuffle(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} records to {args.output}")
    print("families: document_date, profession_surname, location_precision; positive/negative ratio: 1:1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
