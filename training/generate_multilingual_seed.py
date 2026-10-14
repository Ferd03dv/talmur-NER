#!/usr/bin/env python3
"""Generate initial multilingual seed examples for Talos NER.

These records are intentionally small: they are reference templates for the
dataset-generation workflow, not a production corpus. They keep DeepMount label
names so the first multilingual student remains extension-compatible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ent(surface, label, canonical, sensitivity="high", action="mask", tags=None):
    return {
        "surface": surface,
        "label": label,
        "canonical_entity": canonical,
        "sensitivity": sensitivity,
        "action": action,
        "context_tags": tags or [],
    }


RECORDS = [
    {
        "id": "en-seed-0001",
        "language": "en",
        "split": "train",
        "source": {"kind": "synthetic_seed", "template_id": "en_customer_identity_v1", "document_type": "customer_record"},
        "text": "Customer: Alice Brown, born on 1988-05-14, passport PA998877, address 221 Baker Street, London SW1A 1AA, United Kingdom, email alice.brown@example.com, phone +44 7700 900123.",
        "entities": [
            ent("Alice", "NOME", "PERSON_FIRSTNAME", tags=["customer_record"]),
            ent("Brown", "COGNOME", "PERSON_LASTNAME", tags=["customer_record"]),
            ent("1988-05-14", "DATA_NASCITA", "DATE_OF_BIRTH", tags=["date_of_birth"]),
            ent("PA998877", "NUMERO_DOCUMENTO", "DOCUMENT_NUMBER", "critical", tags=["identity_document"]),
            ent("221 Baker Street", "INDIRIZZO", "ADDRESS", tags=["address"]),
            ent("London", "LUOGO", "LOCATION", "medium", tags=["city"]),
            ent("SW1A 1AA", "CODICE_POSTALE", "POSTAL_CODE", "medium", tags=["postal_code"]),
            ent("United Kingdom", "STATO", "COUNTRY", "low", "review", ["country"]),
            ent("alice.brown@example.com", "EMAIL", "EMAIL_ADDRESS", tags=["contact"]),
            ent("+44 7700 900123", "NUMERO_TELEFONO", "PHONE_NUMBER", tags=["contact"]),
        ],
        "record_tags": ["multilingual", "identity", "contact"],
    },
    {
        "id": "en-seed-0002",
        "language": "en",
        "split": "train",
        "source": {"kind": "synthetic_seed", "template_id": "en_finance_secret_v1", "document_type": "finance_note"},
        "text": "Vendor Northwind Ltd uses IBAN GB82WEST12345698765432, BIC BARCGB22, account 44332211. Invoice amount 12,400.00 GBP. Temporary password is RedFox-2026!.",
        "entities": [
            ent("Northwind Ltd", "RAGIONE_SOCIALE", "ORGANIZATION", "medium", tags=["vendor"]),
            ent("GB82WEST12345698765432", "IBAN", "IBAN_CODE", "critical", tags=["bank_account"]),
            ent("BARCGB22", "BIC", "SWIFT_CODE", tags=["bank_identifier"]),
            ent("44332211", "NUMERO_CONTO", "ACCOUNT_NUMBER", "critical", tags=["bank_account"]),
            ent("12,400.00", "IMPORTO", "AMOUNT", "medium", tags=["invoice_amount"]),
            ent("GBP", "VALUTA", "CURRENCY", "low", "review", ["currency"]),
            ent("RedFox-2026!", "PASSWORD", "PASSWORD", "critical", tags=["secret"]),
        ],
        "record_tags": ["multilingual", "financial", "secret"],
    },
    {
        "id": "fr-seed-0001",
        "language": "fr",
        "split": "train",
        "source": {"kind": "synthetic_seed", "template_id": "fr_health_v1", "document_type": "medical_note"},
        "text": "Dossier medical FR-CC-2026-77: la patiente Claire Martin presente une hypertension. Traitement: Metformine 500 mg, comprime, deux fois par jour pendant 30 jours.",
        "entities": [
            ent("FR-CC-2026-77", "CARTELLA_CLINICA", "MEDICAL_RECORD", "critical", tags=["medical_record"]),
            ent("Claire", "NOME", "PERSON_FIRSTNAME", tags=["patient"]),
            ent("Martin", "COGNOME", "PERSON_LASTNAME", tags=["patient"]),
            ent("hypertension", "MALATTIA", "DISEASE", "critical", tags=["disease"]),
            ent("Metformine", "MEDICINA", "MEDICATION", tags=["medication"]),
            ent("500 mg", "DOSAGGIO", "DOSAGE", tags=["dosage"]),
            ent("comprime", "FORM", "DRUG_FORM", "medium", tags=["drug_form"]),
            ent("deux fois par jour", "FREQUENZA", "FREQUENCY", tags=["frequency"]),
            ent("30 jours", "DURATION", "DURATION", "medium", tags=["duration"]),
        ],
        "record_tags": ["multilingual", "health"],
    },
    {
        "id": "fr-seed-0002",
        "language": "fr",
        "split": "validation",
        "source": {"kind": "synthetic_seed", "template_id": "fr_benign_amount_v1", "document_type": "market_note"},
        "text": "Le marche public est estime a 5 milliards d'euros selon une source ouverte.",
        "entities": [
            ent("5 milliards", "IMPORTO", "AMOUNT", "low", "keep", ["public_market_size"]),
            ent("euros", "VALUTA", "CURRENCY", "none", "keep", ["public_market_size"]),
        ],
        "record_tags": ["multilingual", "benign_context", "amount"],
    },
    {
        "id": "de-seed-0001",
        "language": "de",
        "split": "train",
        "source": {"kind": "synthetic_seed", "template_id": "de_legal_v1", "document_type": "legal_memo"},
        "text": "Rechtsanwalt Thomas Weber reicht beim Gericht Berlin einen Antrag nach Gesetz 241/1990 ein. Urteil Nr. 55/2026 und Lizenz DE-LIC-778 werden erwaehnt.",
        "entities": [
            ent("Rechtsanwalt Thomas Weber", "AVV_NOTAIO", "LAWYER_NOTARY", "medium", tags=["legal_professional"]),
            ent("Gericht Berlin", "TRIBUNALE", "COURT", "medium", tags=["court"]),
            ent("Gesetz 241/1990", "LEGGE", "LAW", "medium", tags=["legal_reference"]),
            ent("Nr. 55/2026", "N_SENTENZA", "JUDGMENT_NUMBER", tags=["judgment_number"]),
            ent("DE-LIC-778", "N_LICENZA", "LICENSE_NUMBER", tags=["license_number"]),
        ],
        "record_tags": ["multilingual", "legal"],
    },
    {
        "id": "de-seed-0002",
        "language": "de",
        "split": "train",
        "source": {"kind": "synthetic_seed", "template_id": "de_device_v1", "document_type": "asset_inventory"},
        "text": "Inventar: IMEI 490154203237518, IP 10.0.0.5, IPv6 2001:db8:3333:4444:5555:6666:7777:8888, MAC AA:BB:CC:DD:EE:FF, URL https://portal.beispiel.de/login.",
        "entities": [
            ent("490154203237518", "IMEI", "IMEI", tags=["device_id"]),
            ent("10.0.0.5", "IP", "IP_ADDRESS", "medium", tags=["network"]),
            ent("2001:db8:3333:4444:5555:6666:7777:8888", "IPV6_1", "IP_ADDRESS", "medium", tags=["network"]),
            ent("AA:BB:CC:DD:EE:FF", "MAC", "MAC_ADDRESS", "medium", tags=["network"]),
            ent("https://portal.beispiel.de/login", "URL", "URL", "medium", tags=["url"]),
        ],
        "record_tags": ["multilingual", "device", "network"],
    },
    {
        "id": "es-seed-0001",
        "language": "es",
        "split": "train",
        "source": {"kind": "synthetic_seed", "template_id": "es_hr_v1", "document_type": "hr_record"},
        "text": "Ficha RRHH: Lucia Garcia, profesion ingeniera de datos, estado civil casada, numero de documento DNI-778899.",
        "entities": [
            ent("Lucia", "NOME", "PERSON_FIRSTNAME", tags=["hr_record"]),
            ent("Garcia", "COGNOME", "PERSON_LASTNAME", tags=["hr_record"]),
            ent("ingeniera de datos", "PROFESSIONE", "PROFESSION", "medium", tags=["profession"]),
            ent("casada", "STATO_CIVILE", "MARITAL_STATUS", tags=["marital_status"]),
            ent("DNI-778899", "NUMERO_DOCUMENTO", "DOCUMENT_NUMBER", "critical", tags=["identity_document"]),
        ],
        "record_tags": ["multilingual", "hr", "identity"],
    },
    {
        "id": "es-seed-0002",
        "language": "es",
        "split": "train",
        "source": {"kind": "synthetic_seed", "template_id": "es_business_payment_v1", "document_type": "invoice"},
        "text": "Proveedor Soluciones Norte S.L., NIF B12345678, banco Santander, cuenta ES9121000418450200051332, importe 8.900,00 EUR.",
        "entities": [
            ent("Soluciones Norte S.L.", "RAGIONE_SOCIALE", "ORGANIZATION", "medium", tags=["supplier"]),
            ent("B12345678", "P_IVA", "IT_VAT_NUMBER", tags=["tax_id_localized"]),
            ent("Santander", "BANCA", "BANK_NAME", "medium", tags=["bank"]),
            ent("ES9121000418450200051332", "IBAN", "IBAN_CODE", "critical", tags=["bank_account"]),
            ent("8.900,00", "IMPORTO", "AMOUNT", "medium", tags=["invoice_amount"]),
            ent("EUR", "VALUTA", "CURRENCY", "low", "review", ["currency"]),
        ],
        "record_tags": ["multilingual", "financial", "business"],
    },
    {
        "id": "pt-seed-0001",
        "language": "pt",
        "split": "train",
        "source": {"kind": "synthetic_seed", "template_id": "pt_identity_v1", "document_type": "customer_record"},
        "text": "Cliente: Ana Costa, nascida em 1992-09-03, morada Rua Augusta 10, Lisboa 1100-053, Portugal, email ana.costa@example.pt.",
        "entities": [
            ent("Ana", "NOME", "PERSON_FIRSTNAME", tags=["customer_record"]),
            ent("Costa", "COGNOME", "PERSON_LASTNAME", tags=["customer_record"]),
            ent("1992-09-03", "DATA_NASCITA", "DATE_OF_BIRTH", tags=["date_of_birth"]),
            ent("Rua Augusta 10", "INDIRIZZO", "ADDRESS", tags=["address"]),
            ent("Lisboa", "LUOGO", "LOCATION", "medium", tags=["city"]),
            ent("1100-053", "CODICE_POSTALE", "POSTAL_CODE", "medium", tags=["postal_code"]),
            ent("Portugal", "STATO", "COUNTRY", "low", "review", ["country"]),
            ent("ana.costa@example.pt", "EMAIL", "EMAIL_ADDRESS", tags=["contact"]),
        ],
        "record_tags": ["multilingual", "identity", "address"],
    },
    {
        "id": "pt-seed-0002",
        "language": "pt",
        "split": "validation",
        "source": {"kind": "synthetic_seed", "template_id": "pt_placeholder_person_v1", "document_type": "prompt"},
        "text": "No exemplo, Joao Silva e apenas um placeholder e nao representa um cliente real.",
        "entities": [
            ent("Joao", "NOME", "PERSON_FIRSTNAME", "low", "keep", ["placeholder"]),
            ent("Silva", "COGNOME", "PERSON_LASTNAME", "low", "keep", ["placeholder"]),
        ],
        "record_tags": ["multilingual", "benign_context", "person"],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in RECORDS:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(RECORDS)} multilingual seed records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
