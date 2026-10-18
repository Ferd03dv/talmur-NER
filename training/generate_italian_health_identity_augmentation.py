#!/usr/bin/env python3
"""Create Italian clinical and identity records for the DeepMount-compatible NER.

The corpus is deliberately fictitious.  It teaches two boundaries that the
existing model misses in documents:

* first-name / surname in both normal and registry order, including uppercase;
* patient-specific clinical information versus similarly shaped non-patient
  educational, catalog, technical, or administrative text.

It does not replace a licensed medical corpus or a human gold set.  It is a
targeted augmentation set to merge with the licensed/base training data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CANONICAL = {
    "NOME": "PERSON_FIRSTNAME",
    "COGNOME": "PERSON_LASTNAME",
    "MALATTIA": "DISEASE",
    "DOSAGGIO": "DOSAGE",
    "FORM": "DRUG_FORM",
    "STRENGTH": "DRUG_STRENGTH",
    "DURATION": "DURATION",
    "FREQUENZA": "FREQUENCY",
    "CARTELLA_CLINICA": "MEDICAL_RECORD",
    "MEDICINA": "MEDICATION",
}

FIRST_NAMES = ("Marco", "Giulia", "Elena", "Luca", "Sara", "Davide", "Anna", "Paolo")
SURNAMES = ("Visione", "Rossi", "Bianchi", "Esposito", "De Luca", "Ferrari", "Romano", "Conti")
DISEASES = (
    "diabete di tipo 2", "ipertensione arteriosa", "asma bronchiale",
    "emicrania cronica", "ipotiroidismo", "dermatite atopica",
    "insufficienza cardiaca", "artrite reumatoide",
)
MEDICATIONS = (
    "metformina", "amoxicillina", "ibuprofene", "atorvastatina",
    "levotiroxina", "salbutamolo", "omeprazolo", "cetirizina",
)
# Keep dose surfaces distinct from DRUG_FORM surfaces.  A single record may
# contain both labels, so values such as "una compressa" would overlap the
# FORM span "compressa" and make BIO supervision ambiguous.
DOSAGES = ("500 mg", "250 mg", "1000 mg", "10 ml", "20 gocce", "1 ml", "2 ml")
FORMS = (
    "compressa", "compresse rivestite", "capsule rigide", "soluzione orale",
    "sospensione orale", "sciroppo", "gocce orali", "cerotto transdermico",
)
STRENGTHS = ("5 mg/ml", "250 mg/5 ml", "0,5 mg/ml", "20 microgrammi/ml", "1%", "10 mg/g")
DURATIONS = ("per 7 giorni", "per 10 giorni", "per 2 settimane", "per un mese", "per 3 mesi")
FREQUENCIES = ("una volta al giorno", "due volte al giorno", "tre volte al giorno", "ogni 8 ore", "al bisogno")


def split_for(index: int, validation_ratio: float) -> str:
    return "validation" if index % round(1 / validation_ratio) == 0 else "train"


def entity(surface: str, label: str, *tags: str) -> dict:
    return {
        "surface": surface,
        "label": label,
        "canonical_entity": CANONICAL[label],
        "sensitivity": "high",
        "action": "mask",
        "context_tags": ["italian_health_identity_v2", *tags],
    }


def record(record_id: str, split: str, text: str, entities: list[dict], family: str, polarity: str) -> dict:
    return {
        "id": record_id,
        "language": "it",
        "split": split,
        "source": {
            "kind": "synthetic_clinical_identity_augmentation",
            "template_id": family,
            "document_type": "clinical_record" if polarity == "positive" else "non_patient_contrast",
            "notes": "Fictitious record. No source-document values or real patient data.",
        },
        "text": text,
        "entities": entities,
        "record_tags": ["italian_health_identity_v2", family, polarity, "it"],
    }


def identity_fragment(first: str, last: str, variant: int) -> tuple[str, list[dict]]:
    """Return registry and natural orders with exact label surfaces."""
    cases = (
        (f"Paziente: {first} {last}.", [(first, "NOME"), (last, "COGNOME")]),
        (f"Cognome e nome: {last} {first}.", [(last, "COGNOME"), (first, "NOME")]),
        (f"Assistito: {last}, {first}.", [(last, "COGNOME"), (first, "NOME")]),
        (f"GENERALITÀ\nCOGNOME NOME\n{last.upper()} {first.upper()}", [(last.upper(), "COGNOME"), (first.upper(), "NOME")]),
        (f"Scheda paziente — {first.upper()} {last.upper()}.", [(first.upper(), "NOME"), (last.upper(), "COGNOME")]),
        (f"Nome: {first}\nCognome: {last}", [(first, "NOME"), (last, "COGNOME")]),
    )
    text, raw = cases[variant % len(cases)]
    return text, [entity(surface, label, "identity", "name_order") for surface, label in raw]


def clinical_record(index: int, rng: random.Random, validation_ratio: float) -> dict:
    first, last = rng.choice(FIRST_NAMES), rng.choice(SURNAMES)
    disease, medication = rng.choice(DISEASES), rng.choice(MEDICATIONS)
    dosage, form = rng.choice(DOSAGES), rng.choice(FORMS)
    strength, duration, frequency = rng.choice(STRENGTHS), rng.choice(DURATIONS), rng.choice(FREQUENCIES)
    chart = f"cartella n. {2024 + index % 3}-{1000 + index:04d}"
    identity, identity_entities = identity_fragment(first, last, index)
    templates = (
        "{identity}\n{chart}. Diagnosi: {disease}.\nTerapia: {medication} {strength}, {form}; assumere {dosage}, {frequency}, {duration}.",
        "{chart}\n{identity}\nPiano terapeutico per {disease}: {medication}, {form} {strength}. Posologia: {dosage}, {frequency}, {duration}.",
        "Estratto {chart} — {identity}\nCondizione clinica: {disease}. Farmaco prescritto: {medication} ({form}, {strength}); dose {dosage}, {frequency}, {duration}.",
    )
    text = rng.choice(templates).format(
        identity=identity, chart=chart, disease=disease, medication=medication,
        dosage=dosage, form=form, strength=strength, duration=duration, frequency=frequency,
    )
    return record(
        f"health-identity-clinical-{index:05d}",
        split_for(index, validation_ratio),
        text,
        identity_entities + [
            entity(chart, "CARTELLA_CLINICA", "patient_record"),
            entity(disease, "MALATTIA", "diagnosis"),
            entity(medication, "MEDICINA", "prescription"),
            entity(strength, "STRENGTH", "prescription"),
            entity(form, "FORM", "prescription"),
            entity(dosage, "DOSAGGIO", "prescription"),
            entity(frequency, "FREQUENZA", "prescription"),
            entity(duration, "DURATION", "prescription"),
        ],
        "clinical_regimen_v2",
        "positive",
    )


def identity_record(index: int, rng: random.Random, validation_ratio: float) -> dict:
    # The first fixtures directly cover the two manual regression strings.
    if index == 1:
        first, last, variant = "Marco", "Visione", 0
    elif index == 2:
        first, last, variant = "Marco", "Visione", 1
    else:
        first, last, variant = rng.choice(FIRST_NAMES), rng.choice(SURNAMES), index
    text, entities = identity_fragment(first, last, variant)
    return record(
        f"health-identity-person-{index:05d}", split_for(index, validation_ratio), text,
        entities, "person_name_order_v2", "positive",
    )


def focused_health_record(index: int, rng: random.Random, validation_ratio: float) -> dict:
    label, values, templates = rng.choice((
        ("MALATTIA", DISEASES, ("Il paziente è in cura per {value}.", "Diagnosi confermata: {value}.")),
        ("MEDICINA", MEDICATIONS, ("Farmaco prescritto: {value}.", "La terapia del paziente include {value}.")),
        ("DOSAGGIO", DOSAGES, ("Dose prescritta: {value} dopo i pasti.", "Assumere {value} secondo ricetta.")),
        ("FORM", FORMS, ("Forma farmaceutica prescritta: {value}.", "Il medicinale è fornito come {value}.")),
        ("STRENGTH", STRENGTHS, ("Concentrazione del farmaco: {value}.", "Il flacone prescritto contiene {value}.")),
        ("DURATION", DURATIONS, ("La terapia deve proseguire {value}.", "Trattamento previsto {value}.")),
        ("FREQUENZA", FREQUENCIES, ("Assumere il medicinale {value}.", "Frequenza della terapia: {value}.")),
        ("CARTELLA_CLINICA", (f"cartella n. 2026-{index:04d}",), ("Aprire la {value}.", "Aggiornamento della {value}.")),
    ))
    value = rng.choice(values)
    return record(
        f"health-identity-focused-{index:05d}", split_for(index, validation_ratio),
        rng.choice(templates).format(value=value), [entity(value, label, "focused_health")],
        f"focused_{label.lower()}_v2", "positive",
    )


def hard_negative(index: int, rng: random.Random, validation_ratio: float) -> dict:
    disease, medication = rng.choice(DISEASES), rng.choice(MEDICATIONS)
    strength, dosage = rng.choice(STRENGTHS), rng.choice(DOSAGES)
    templates = (
        f"Il corso di medicina discute {disease} e {medication} in una bibliografia generale, senza riferimenti a pazienti.",
        f"Catalogo di laboratorio: soluzione di prova {strength}; il valore è tecnico e non è una prescrizione clinica.",
        f"Manuale operativo: il test viene eseguito {rng.choice(FREQUENCIES)} per verificare il servizio, non una terapia.",
        f"La pratica n. {2024 + index % 3}-{1000 + index:04d} riguarda una richiesta amministrativa, non una cartella clinica.",
        f"Il progetto dura {rng.choice(DURATIONS).replace('per ', '')}; non contiene dati sanitari né prescrizioni.",
        f"Il reparto imballa {dosage} di materiale campione; non è una dose destinata a un paziente.",
        "PROGETTO LUMEN ORBITA: titolo interno di una presentazione, non anagrafica di una persona.",
        "La parola compressa descrive un archivio ZIP; non indica una forma farmaceutica né una terapia.",
    )
    return record(
        f"health-identity-negative-{index:05d}", split_for(index, validation_ratio),
        rng.choice(templates), [], "clinical_identity_hard_negative_v2", "hard_negative",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clinical-records", type=int, default=800)
    parser.add_argument("--identity-records", type=int, default=600)
    parser.add_argument("--focused-health-records", type=int, default=800)
    parser.add_argument("--hard-negatives", type=int, default=800)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    counts = (args.clinical_records, args.identity_records, args.focused_health_records, args.hard_negatives)
    if min(counts) < 1 or not 0 < args.validation_ratio < 1:
        raise SystemExit("all record counts must be positive and validation-ratio must be between 0 and 1")

    rng = random.Random(args.seed)
    records = []
    records.extend(clinical_record(i, rng, args.validation_ratio) for i in range(1, args.clinical_records + 1))
    records.extend(identity_record(i, rng, args.validation_ratio) for i in range(1, args.identity_records + 1))
    records.extend(focused_health_record(i, rng, args.validation_ratio) for i in range(1, args.focused_health_records + 1))
    records.extend(hard_negative(i, rng, args.validation_ratio) for i in range(1, args.hard_negatives + 1))
    rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for item in records:
            out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} records to {args.output}")
    print("labels: NOME, COGNOME, MALATTIA, MEDICINA, DOSAGGIO, FORM, STRENGTH, DURATION, FREQUENZA, CARTELLA_CLINICA")
    print("families: clinical_regimen=positive, person_name_order=positive, focused_health=positive, clinical_identity_hard_negative=O-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
