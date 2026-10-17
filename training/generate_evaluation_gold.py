#!/usr/bin/env python3
"""Create a deterministic, test-only gold set for labels absent from OpenPII.

This generator is deliberately separate from training augmentation.  It uses
different sentence forms, values and hard negatives.  Its output must never be
merged into a training source JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TARGET_LABELS = (
    "CVV", "DOSAGGIO", "DURATION", "FREQUENZA", "MALATTIA", "MEDICINA",
    "ORARIO", "PASSWORD", "PIN", "PROFESSIONE", "STATO", "STATO_CIVILE",
    "TARGA_VEICOLO",
)

CANONICAL = {
    "CVV": "CARD_CVV", "DOSAGGIO": "DOSAGE", "DURATION": "DURATION",
    "FREQUENZA": "FREQUENCY", "MALATTIA": "DISEASE", "MEDICINA": "MEDICATION",
    "ORARIO": "TIME", "PASSWORD": "PASSWORD", "PIN": "DEVICE_PIN",
    "PROFESSIONE": "PROFESSION", "STATO": "COUNTRY",
    "STATO_CIVILE": "MARITAL_STATUS", "TARGA_VEICOLO": "VEHICLE_PLATE",
}

# The six language variants are intentionally short document fragments.  They
# do not reuse the training generator's templates.
POSITIVE = {
    "it": {
        "CVV": "Verifica pagamento — codice sul retro: {value}.", "DOSAGGIO": "Schema terapeutico aggiornato: {value}.",
        "DURATION": "Ciclo previsto: {value}.", "FREQUENZA": "Somministrazione programmata {value}.",
        "MALATTIA": "Diagnosi confermata: {value}.", "MEDICINA": "Principio attivo indicato: {value}.",
        "ORARIO": "Orario della visita: {value}.", "PASSWORD": "Chiave di accesso provvisoria = {value}.",
        "PIN": "Codice di sblocco del dispositivo: {value}.", "PROFESSIONE": "Qualifica nel curriculum: {value}.",
        "STATO": "Nazionalità dichiarata: {value}.", "STATO_CIVILE": "Condizione familiare dichiarata: {value}.",
        "TARGA_VEICOLO": "Veicolo registrato con identificativo {value}.",
    },
    "en": {
        "CVV": "Payment check — code on the back: {value}.", "DOSAGGIO": "Updated treatment schedule: {value}.",
        "DURATION": "Planned course: {value}.", "FREQUENZA": "Administration scheduled {value}.",
        "MALATTIA": "Confirmed diagnosis: {value}.", "MEDICINA": "Active medicine listed: {value}.",
        "ORARIO": "Appointment time: {value}.", "PASSWORD": "Temporary access key = {value}.",
        "PIN": "Device unlock code: {value}.", "PROFESSIONE": "CV qualification: {value}.",
        "STATO": "Declared nationality: {value}.", "STATO_CIVILE": "Declared marital status: {value}.",
        "TARGA_VEICOLO": "Registered vehicle identifier {value}.",
    },
    "fr": {
        "CVV": "Contrôle du paiement — code au dos : {value}.", "DOSAGGIO": "Schéma thérapeutique mis à jour : {value}.",
        "DURATION": "Durée prévue : {value}.", "FREQUENZA": "Administration prévue {value}.",
        "MALATTIA": "Diagnostic confirmé : {value}.", "MEDICINA": "Médicament actif indiqué : {value}.",
        "ORARIO": "Heure du rendez-vous : {value}.", "PASSWORD": "Clé d'accès temporaire = {value}.",
        "PIN": "Code de déverrouillage : {value}.", "PROFESSIONE": "Qualification dans le CV : {value}.",
        "STATO": "Nationalité déclarée : {value}.", "STATO_CIVILE": "Situation familiale déclarée : {value}.",
        "TARGA_VEICOLO": "Identifiant du véhicule enregistré : {value}.",
    },
    "de": {
        "CVV": "Zahlungsprüfung — Code auf der Rückseite: {value}.", "DOSAGGIO": "Aktualisierter Therapieplan: {value}.",
        "DURATION": "Geplante Behandlungsdauer: {value}.", "FREQUENZA": "Einnahme vorgesehen {value}.",
        "MALATTIA": "Bestätigte Diagnose: {value}.", "MEDICINA": "Angegebenes Arzneimittel: {value}.",
        "ORARIO": "Terminzeit: {value}.", "PASSWORD": "Temporärer Zugangsschlüssel = {value}.",
        "PIN": "Geräte-Entsperrcode: {value}.", "PROFESSIONE": "Berufsbezeichnung im Lebenslauf: {value}.",
        "STATO": "Angegebene Staatsangehörigkeit: {value}.", "STATO_CIVILE": "Angegebener Familienstand: {value}.",
        "TARGA_VEICOLO": "Registriertes Fahrzeugkennzeichen: {value}.",
    },
    "es": {
        "CVV": "Comprobación de pago — código trasero: {value}.", "DOSAGGIO": "Pauta terapéutica actualizada: {value}.",
        "DURATION": "Duración prevista: {value}.", "FREQUENZA": "Administración programada {value}.",
        "MALATTIA": "Diagnóstico confirmado: {value}.", "MEDICINA": "Medicamento activo indicado: {value}.",
        "ORARIO": "Hora de la cita: {value}.", "PASSWORD": "Clave temporal de acceso = {value}.",
        "PIN": "Código de desbloqueo del dispositivo: {value}.", "PROFESSIONE": "Profesión en el currículo: {value}.",
        "STATO": "Nacionalidad declarada: {value}.", "STATO_CIVILE": "Estado civil declarado: {value}.",
        "TARGA_VEICOLO": "Identificador del vehículo registrado: {value}.",
    },
    "pt": {
        "CVV": "Verificação de pagamento — código no verso: {value}.", "DOSAGGIO": "Plano terapêutico atualizado: {value}.",
        "DURATION": "Duração prevista: {value}.", "FREQUENZA": "Administração prevista {value}.",
        "MALATTIA": "Diagnóstico confirmado: {value}.", "MEDICINA": "Medicamento ativo indicado: {value}.",
        "ORARIO": "Hora da consulta: {value}.", "PASSWORD": "Chave temporária de acesso = {value}.",
        "PIN": "Código de desbloqueio do dispositivo: {value}.", "PROFESSIONE": "Profissão no currículo: {value}.",
        "STATO": "Nacionalidade declarada: {value}.", "STATO_CIVILE": "Estado civil declarado: {value}.",
        "TARGA_VEICOLO": "Identificador do veículo registado: {value}.",
    },
}

NEGATIVE = {
    "it": ("Il codice di verifica non è stato richiesto.", "Il documento descrive una procedura generale.", "Il campo resta vuoto in questa copia."),
    "en": ("No verification code was requested.", "The document describes a general procedure.", "The field remains empty in this copy."),
    "fr": ("Aucun code de vérification n'a été demandé.", "Le document décrit une procédure générale.", "Le champ reste vide dans cette copie."),
    "de": ("Es wurde kein Bestätigungscode angefordert.", "Das Dokument beschreibt ein allgemeines Verfahren.", "Das Feld bleibt in dieser Kopie leer."),
    "es": ("No se solicitó ningún código de verificación.", "El documento describe un procedimiento general.", "El campo queda vacío en esta copia."),
    "pt": ("Não foi solicitado nenhum código de verificação.", "O documento descreve um procedimento geral.", "O campo fica vazio nesta cópia."),
}

REFERENCE = {"it": "Riferimento", "en": "Reference", "fr": "Référence", "de": "Referenz", "es": "Referencia", "pt": "Referência"}

VALUES = {
    "CVV": ("042", "718", "905"), "DOSAGGIO": ("375 mg", "1,5 ml", "2 compresse"),
    "DURATION": ("undici giorni", "tre settimane", "48 ore"), "FREQUENZA": ("ogni otto ore", "due volte al giorno", "la sera"),
    "MALATTIA": ("asma bronchiale", "diabete di tipo 2", "ipertensione arteriosa"),
    "MEDICINA": ("metformina", "amoxicillina", "atorvastatina"), "ORARIO": ("07:45", "18:30", "23:10"),
    "PASSWORD": ("Faro!27-Luna", "Menta_48#Blu", "Rete$2026Sicura"), "PIN": ("4829", "1607", "9354"),
    "PROFESSIONE": ("architetta", "tecnico di laboratorio", "consulente legale"),
    "STATO": ("Italia", "Portogallo", "Germania"), "STATO_CIVILE": ("coniugata", "divorziato", "unita civilmente"),
    "TARGA_VEICOLO": ("ZX 418 PL", "GH 729 RM", "VK 063 TN"),
}


def entity(value: str, label: str) -> dict:
    return {"surface": value, "label": label, "canonical_entity": CANONICAL[label], "sensitivity": "critical" if label in {"CVV", "PASSWORD", "PIN"} else "high", "action": "mask"}


def record(record_id: str, language: str, text: str, entities: list[dict], label: str, polarity: str, seed: int) -> dict:
    return {
        "id": record_id, "language": language, "split": "evaluation", "text": text, "entities": entities,
        "source": {"kind": "generated_evaluation_gold", "template_id": f"heldout_{label.lower()}_{polarity}_v1", "notes": "Test-only data. Never merge into training.", "seed": seed},
        "record_tags": ["evaluation_gold_v1", polarity, language, label],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--languages", default="it,en,fr,de,es,pt")
    parser.add_argument("--records-per-label", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    if args.records_per_label < 2 or args.records_per_label % 2:
        raise SystemExit("--records-per-label must be an even number of at least 2")
    languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
    unknown = set(languages) - set(POSITIVE)
    if unknown:
        raise SystemExit("unsupported languages: " + ", ".join(sorted(unknown)))

    rng = random.Random(args.seed)
    half = args.records_per_label // 2
    rows: list[dict] = []
    for language in languages:
        for label in TARGET_LABELS:
            for index in range(half):
                value = rng.choice(VALUES[label])
                rows.append(record(f"gold-{language}-{label.lower()}-positive-{index + 1:03d}", language, POSITIVE[language][label].format(value=value), [entity(value, label)], label, "positive", args.seed))
                # Never include the label name in a negative: doing so would
                # turn an evaluation hard-negative into a label-word cue.
                negative = f"{rng.choice(NEGATIVE[language])} {REFERENCE[language]} {index + 1:05d}."
                rows.append(record(f"gold-{language}-{label.lower()}-negative-{index + 1:03d}", language, negative, [], label, "hard_negative", args.seed))
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for item in rows:
            out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} evaluation-only records to {args.output}")
    print(f"languages={','.join(languages)}; labels={len(TARGET_LABELS)}; records per label/language={args.records_per_label}; seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
