#!/usr/bin/env python3
"""Generate a large synthetic source JSONL corpus for Talmur NER.

This is CPU-only. It creates source records with entity surfaces, not tokenized
tensors. Run `compile_jsonl_to_bio.py` afterwards to resolve offsets and catch
overlaps before any GPU training job.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


FIRST = {
    "it": ["Giulia", "Marco", "Elena", "Ferdinando", "Laura", "Matteo", "Sara", "Luca"],
    "en": ["Alice", "Oliver", "Emma", "George", "Charlotte", "Henry", "Grace", "Noah"],
    "fr": ["Claire", "Louis", "Camille", "Hugo", "Manon", "Jules", "Lea", "Nathan"],
    "de": ["Anna", "Thomas", "Lena", "Felix", "Marie", "Jonas", "Laura", "Max"],
    "es": ["Lucia", "Carlos", "Sofia", "Diego", "Marta", "Javier", "Elena", "Pablo"],
    "pt": ["Ana", "Joao", "Maria", "Pedro", "Ines", "Miguel", "Sofia", "Tiago"],
}

LAST = {
    "it": ["Bianchi", "Rossi", "Conti", "Del Vecchio", "Neri", "Greco", "Romano", "Ferrari"],
    "en": ["Brown", "Smith", "Johnson", "Taylor", "Wilson", "Evans", "Walker", "Harris"],
    "fr": ["Martin", "Bernard", "Dubois", "Moreau", "Laurent", "Leroy", "Roux", "Fournier"],
    "de": ["Weber", "Schmidt", "Muller", "Fischer", "Wagner", "Becker", "Hoffmann", "Schulz"],
    "es": ["Garcia", "Lopez", "Martinez", "Sanchez", "Perez", "Gomez", "Ruiz", "Diaz"],
    "pt": ["Costa", "Silva", "Santos", "Pereira", "Ferreira", "Oliveira", "Ribeiro", "Almeida"],
}

CITY = {
    "it": ["Roma", "Milano", "Torino", "Bologna"],
    "en": ["London", "Manchester", "Bristol", "Leeds"],
    "fr": ["Paris", "Lyon", "Marseille", "Lille"],
    "de": ["Berlin", "Munich", "Hamburg", "Cologne"],
    "es": ["Madrid", "Barcelona", "Valencia", "Sevilla"],
    "pt": ["Lisboa", "Porto", "Coimbra", "Braga"],
}

COUNTRY = {"it": "Italia", "en": "United Kingdom", "fr": "France", "de": "Deutschland", "es": "Espana", "pt": "Portugal"}
STREET = {"it": "Via Roma", "en": "Baker Street", "fr": "Rue Victor Hugo", "de": "Hauptstrasse", "es": "Calle Mayor", "pt": "Rua Augusta"}
ORG_SUFFIX = {"it": "S.r.l.", "en": "Ltd", "fr": "SAS", "de": "GmbH", "es": "S.L.", "pt": "Lda"}
ORG_PREFIX = ["Nova", "Blue", "Green", "North", "Prime", "Vertex", "Atlas", "Orion"]
ORG_NOUN = ["Systems", "Finance", "Health", "Logistics", "Foods", "Data", "Energy", "Consulting"]
BRANDS = ["EcoPress", "MediFlow", "Talos", "NordPay", "AstraDesk", "ClearBox", "FinPilot", "BioTrace"]
BANKS = ["Intesa Sanpaolo", "Santander", "BNP Paribas", "Deutsche Bank", "Barclays", "Caixa Geral"]
DISEASES = ["diabete mellito", "ipertensione", "asma bronchiale", "emicrania cronica", "artrite reumatoide"]
MEDS = ["Metformina", "Amoxicillina", "Ibuprofene", "Atorvastatina", "Lisinopril"]
PROF = ["ingegnere gestionale", "data analyst", "medico", "avvocato", "contabile", "project manager"]
MARITAL = ["coniugata", "single", "separato", "divorziata", "vedovo"]

CANONICAL = {
    "AVV_NOTAIO": "LAWYER_NOTARY", "BANCA": "BANK_NAME", "BIC": "SWIFT_CODE", "BRAND": "BRAND",
    "CARTELLA_CLINICA": "MEDICAL_RECORD", "CODICE_FISCALE": "IT_FISCAL_CODE", "CODICE_POSTALE": "POSTAL_CODE",
    "COGNOME": "PERSON_LASTNAME", "CVV": "CARD_CVV", "DATA": "DATE", "DATA_MORTE": "DATE_OF_DEATH",
    "DATA_NASCITA": "DATE_OF_BIRTH", "DOSAGGIO": "DOSAGE", "DURATION": "DURATION", "EMAIL": "EMAIL_ADDRESS",
    "ETA": "AGE", "FOGLIO": "CADASTRE_SHEET", "FORM": "DRUG_FORM", "FREQUENZA": "FREQUENCY",
    "IBAN": "IBAN_CODE", "IMEI": "IMEI", "IMPORTO": "AMOUNT", "INDIRIZZO": "ADDRESS", "IP": "IP_ADDRESS",
    "IPV6_1": "IP_ADDRESS", "LEGGE": "LAW", "LUOGO": "LOCATION", "MAC": "MAC_ADDRESS", "MALATTIA": "DISEASE",
    "MAPPALE": "CADASTRE_MAP", "MEDICINA": "MEDICATION", "NOME": "PERSON_FIRSTNAME", "NUMERO_CARTA": "CREDIT_CARD",
    "NUMERO_CONTO": "ACCOUNT_NUMBER", "NUMERO_DOCUMENTO": "DOCUMENT_NUMBER", "NUMERO_TELEFONO": "PHONE_NUMBER",
    "NUM_ASSEGNO_BANCARIO": "BANK_CHECK", "N_LICENZA": "LICENSE_NUMBER", "N_SENTENZA": "JUDGMENT_NUMBER",
    "ORARIO": "TIME", "PARTICELLA": "CADASTRE_PARCEL", "PASSWORD": "PASSWORD", "PIN": "DEVICE_PIN",
    "PROFESSIONE": "PROFESSION", "P_IVA": "IT_VAT_NUMBER", "RAGIONE_SOCIALE": "ORGANIZATION",
    "REGIME_PATRIMONIALE": "MATRIMONIAL_REGIME", "STATO": "COUNTRY", "STATO_CIVILE": "MARITAL_STATUS",
    "STORIA_CLINICA": "CLINICAL_HISTORY", "STRENGTH": "DRUG_STRENGTH", "SUBALTERNO": "CADASTRE_SUB",
    "TARGA_VEICOLO": "IT_VEHICLE_PLATE", "TASSO_MUTUO": "MORTGAGE_RATE", "TRIBUNALE": "COURT",
    "URL": "URL", "USER_AGENT": "USER_AGENT", "VALUTA": "CURRENCY",
}


def e(surface, label, sensitivity="high", action="mask", tags=None, occurrence=None):
    entity = {
        "surface": str(surface),
        "label": label,
        "canonical_entity": CANONICAL[label],
        "sensitivity": sensitivity,
        "action": action,
        "context_tags": tags or [],
    }
    if occurrence is not None:
        entity["occurrence"] = occurrence
    return entity


def digits(rng, n):
    return "".join(rng.choice("0123456789") for _ in range(n))


def pick(rng, items):
    return rng.choice(items)


def person(rng, lang):
    return pick(rng, FIRST[lang]), pick(rng, LAST[lang])


def org(rng, lang):
    return f"{pick(rng, ORG_PREFIX)} {pick(rng, ORG_NOUN)} {ORG_SUFFIX[lang]}"


def date(rng):
    return f"{rng.randint(1970, 2026):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def phone(rng, lang):
    prefix = {"it": "+39", "en": "+44", "fr": "+33", "de": "+49", "es": "+34", "pt": "+351"}[lang]
    return f"{prefix} {rng.randint(300, 999)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"


def iban(rng, lang):
    cc = {"it": "IT", "en": "GB", "fr": "FR", "de": "DE", "es": "ES", "pt": "PT"}[lang]
    return f"{cc}{rng.randint(10, 99)}{rng.choice(['X', 'A', 'B', 'C'])}{digits(rng, 22)}"


def template_identity(rng, lang, idx):
    first, last = person(rng, lang)
    city = pick(rng, CITY[lang])
    postal = f"{rng.randint(10000, 99999)}"
    address = f"{STREET[lang]} {rng.randint(1, 220)}"
    doc = f"DOC-{lang.upper()}-{digits(rng, 7)}"
    tax = f"TAX{digits(rng, 13)}"
    mail = f"{first.lower()}.{last.lower().replace(' ', '')}{idx}@example.{lang}"
    text = f"Cliente {first} {last}, nato il {date(rng)}, eta {rng.randint(18, 89)} anni, documento {doc}, codice fiscale {tax}, residente in {address}, {postal} {city}, {COUNTRY[lang]}. Email {mail}, telefono {phone(rng, lang)}."
    return text, [
        e(first, "NOME", tags=["person"]), e(last, "COGNOME", tags=["person"]), e(date(rng), "DATA", "low", "review", ["extra_date_unused"])
    ]


def record_identity(rng, lang, idx):
    first, last = person(rng, lang)
    city = pick(rng, CITY[lang])
    postal = f"{rng.randint(10000, 99999)}"
    address = f"{STREET[lang]} {rng.randint(1, 220)}"
    birth = date(rng)
    age = f"{rng.randint(18, 89)} anni"
    doc = f"DOC-{lang.upper()}-{digits(rng, 7)}"
    tax = f"TAX{digits(rng, 13)}"
    mail = f"{first.lower()}.{last.lower().replace(' ', '')}{idx}@example.{lang}"
    tel = phone(rng, lang)
    text = f"Cliente {first} {last}, nato il {birth}, eta {age}, documento {doc}, codice fiscale {tax}, residente in {address}, {postal} {city}, {COUNTRY[lang]}. Email {mail}, telefono {tel}."
    city_occurrence = 2 if city in address else None
    return text, [e(first, "NOME"), e(last, "COGNOME"), e(birth, "DATA_NASCITA"), e(age, "ETA", "medium"), e(doc, "NUMERO_DOCUMENTO", "critical"), e(tax, "CODICE_FISCALE", "critical"), e(address, "INDIRIZZO"), e(postal, "CODICE_POSTALE", "medium"), e(city, "LUOGO", "medium", occurrence=city_occurrence), e(COUNTRY[lang], "STATO", "low", "review"), e(mail, "EMAIL"), e(tel, "NUMERO_TELEFONO")]


def record_finance(rng, lang, idx):
    company = org(rng, lang)
    bank = pick(rng, BANKS)
    vat = f"VAT{digits(rng, 11)}"
    ib = iban(rng, lang)
    bic = f"{rng.choice(['BCIT', 'BARC', 'DEUT', 'BNPA'])}{lang.upper()}{rng.randint(10,99)}"
    account = f"AC{digits(rng, 10)}"
    amount = f"{rng.randint(1, 90)}.{rng.randint(100, 999)},00"
    currency = pick(rng, ["EUR", "GBP", "USD"])
    brand = pick(rng, BRANDS)
    text = f"Fornitore {company}, P. IVA {vat}, banca {bank}, IBAN {ib}, BIC {bic}, conto {account}. Importo fattura {amount} {currency} per il brand {brand}."
    return text, [e(company, "RAGIONE_SOCIALE", "medium"), e(vat, "P_IVA"), e(bank, "BANCA", "medium"), e(ib, "IBAN", "critical"), e(bic, "BIC"), e(account, "NUMERO_CONTO", "critical"), e(amount, "IMPORTO", "medium"), e(currency, "VALUTA", "low", "review"), e(brand, "BRAND", "low", "review")]


def record_payment_secret(rng, lang, idx):
    card = f"4{digits(rng, 15)}"
    cvv = digits(rng, 3)
    pin = digits(rng, 4)
    password = f"Pwd-{pick(rng, BRANDS)}-{rng.randint(100,999)}!"
    text = f"Non condividere carta {card}, CVV {cvv}, PIN {pin} e password temporanea {password}."
    return text, [e(card, "NUMERO_CARTA", "critical"), e(cvv, "CVV", "critical"), e(pin, "PIN", "critical"), e(password, "PASSWORD", "critical")]


def record_legal(rng, lang, idx):
    first, last = person(rng, lang)
    court_city = pick(rng, CITY[lang])
    lawyer = f"avv. {first} {last}"
    court = f"Tribunale di {court_city}"
    law = f"Legge {rng.randint(1,999)}/{rng.randint(1980,2026)}"
    judgment = f"N. {rng.randint(100,9999)}/{rng.randint(2020,2026)}"
    lic = f"LIC-{lang.upper()}-{digits(rng, 4)}"
    regime = pick(rng, ["separazione dei beni", "comunione dei beni", "patto patrimoniale riservato"])
    text = f"{lawyer} deposita ricorso presso il {court} ai sensi della {law}. La sentenza {judgment} cita licenza {lic} e regime patrimoniale di {regime}."
    return text, [e(lawyer, "AVV_NOTAIO", "medium"), e(court, "TRIBUNALE", "medium"), e(law, "LEGGE", "medium"), e(judgment, "N_SENTENZA"), e(lic, "N_LICENZA"), e(regime, "REGIME_PATRIMONIALE")]


def record_cadastre(rng, lang, idx):
    sheet = f"foglio {rng.randint(1, 99)}"
    map_ = f"mappale {rng.randint(100, 999)}"
    parcel = f"particella {rng.randint(1, 200)}"
    sub = f"subalterno {rng.randint(1, 30)}"
    city = pick(rng, CITY[lang])
    text = f"Dati catastali: {sheet}, {map_}, {parcel}, {sub} nel comune di {city}."
    return text, [e(sheet, "FOGLIO", "medium"), e(map_, "MAPPALE", "medium"), e(parcel, "PARTICELLA", "medium"), e(sub, "SUBALTERNO", "medium"), e(city, "LUOGO", "low", "review")]


def record_health(rng, lang, idx):
    first, last = person(rng, lang)
    medrec = f"CC-{lang.upper()}-{rng.randint(2020,2026)}-{digits(rng,3)}"
    disease = pick(rng, DISEASES)
    med = pick(rng, MEDS)
    dosage = f"{pick(rng, [250, 500, 850, 1000])} mg"
    form = pick(rng, ["compressa", "capsula", "sospensione orale", "fiala"])
    freq = pick(rng, ["una volta al giorno", "due volte al giorno", "ogni 8 ore"])
    duration = f"{rng.randint(5, 90)} giorni"
    strength = f"{pick(rng, [125, 250, 500])} mg/5 ml"
    history = f"{pick(rng, DISEASES)} dal {rng.randint(2000,2022)}"
    text = f"Cartella clinica {medrec}: paziente {first} {last} con {disease}. Terapia: {med} {dosage}, {form}, {freq} per {duration}. Strength {strength}. Storia clinica: {history}."
    return text, [e(medrec, "CARTELLA_CLINICA", "critical"), e(first, "NOME"), e(last, "COGNOME"), e(disease, "MALATTIA", "critical"), e(med, "MEDICINA"), e(dosage, "DOSAGGIO"), e(form, "FORM", "medium"), e(freq, "FREQUENZA"), e(duration, "DURATION", "medium"), e(strength, "STRENGTH"), e(history, "STORIA_CLINICA", "critical")]


def record_device(rng, lang, idx):
    imei = digits(rng, 15)
    ip = f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
    ipv6 = f"2001:db8:{rng.randint(1000,9999)}:{rng.randint(1000,9999)}:{rng.randint(1000,9999)}:{rng.randint(1000,9999)}:{rng.randint(1000,9999)}:{rng.randint(1000,9999)}"
    mac = ":".join(f"{rng.randint(0,255):02X}" for _ in range(6))
    plate = f"{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.randint(100,999)}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
    ua = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) TalosSeed/{idx}"
    url = f"https://portal{idx}.example.{lang}/login"
    text = f"Asset: IMEI {imei}, IP {ip}, IPv6 {ipv6}, MAC {mac}, targa {plate}, user agent {ua}, URL {url}."
    return text, [e(imei, "IMEI"), e(ip, "IP", "medium"), e(ipv6, "IPV6_1", "medium"), e(mac, "MAC", "medium"), e(plate, "TARGA_VEICOLO"), e(ua, "USER_AGENT", "medium"), e(url, "URL", "medium")]


def record_bank_event(rng, lang, idx):
    time = f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}"
    event_date = date(rng)
    check = f"CHK-{digits(rng, 8)}"
    rate = f"{rng.randint(1,7)},{rng.randint(10,99)}%"
    death = date(rng)
    text = f"Alle ore {time} del {event_date} registrato assegno bancario {check}, tasso mutuo {rate}, data morte garante {death}."
    return text, [e(time, "ORARIO", "low", "review"), e(event_date, "DATA", "medium"), e(check, "NUM_ASSEGNO_BANCARIO"), e(rate, "TASSO_MUTUO"), e(death, "DATA_MORTE")]


def record_hr(rng, lang, idx):
    first, last = person(rng, lang)
    profession = pick(rng, PROF)
    marital = pick(rng, MARITAL)
    text = f"Scheda HR: {first} {last}, professione {profession}, stato civile {marital}."
    return text, [e(first, "NOME"), e(last, "COGNOME"), e(profession, "PROFESSIONE", "medium"), e(marital, "STATO_CIVILE")]


def record_benign_person(rng, lang, idx):
    first, last = person(rng, lang)
    text = f"Nel documento di esempio {first} {last} e' un placeholder e non rappresenta un cliente reale."
    return text, [e(first, "NOME", "low", "keep", ["placeholder"]), e(last, "COGNOME", "low", "keep", ["placeholder"])]


def record_benign_amount(rng, lang, idx):
    amount = f"{rng.randint(1, 99)} miliardi"
    currency = pick(rng, ["euro", "EUR", "dollari"])
    text = f"Il mercato pubblico vale {amount} di {currency} secondo una fonte aperta."
    return text, [e(amount, "IMPORTO", "low", "keep", ["public_market_size"]), e(currency, "VALUTA", "none", "keep", ["public_market_size"])]


TEMPLATES = [
    record_identity,
    record_finance,
    record_payment_secret,
    record_legal,
    record_cadastre,
    record_health,
    record_device,
    record_bank_event,
    record_hr,
    record_benign_person,
    record_benign_amount,
]


def make_record(rng, idx, lang):
    template = TEMPLATES[idx % len(TEMPLATES)]
    text, entities = template(rng, lang, idx)
    return {
        "id": f"{lang}-synthetic-{idx:08d}",
        "language": lang,
        "split": "validation" if idx % 10 == 0 else "train",
        "source": {
            "kind": "synthetic_generated",
            "template_id": template.__name__,
            "document_type": template.__name__.replace("record_", ""),
        },
        "text": text,
        "entities": entities,
        "record_tags": ["synthetic_v1", lang, template.__name__],
    }


def parse_language_weights(raw):
    weights = []
    for part in raw.split(","):
        lang, weight = part.split(":", 1)
        weights.append((lang.strip(), float(weight)))
    total = sum(w for _, w in weights)
    return [(lang, w / total) for lang, w in weights]


def choose_language(rng, weights):
    x = rng.random()
    acc = 0.0
    for lang, weight in weights:
        acc += weight
        if x <= acc:
            return lang
    return weights[-1][0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language-weights", default="it:40,en:25,fr:20,de:20,es:25,pt:20")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    weights = parse_language_weights(args.language_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = {lang: 0 for lang, _ in weights}
    with args.output.open("w", encoding="utf-8") as f:
        for idx in range(1, args.records + 1):
            lang = choose_language(rng, weights)
            counts[lang] += 1
            record = make_record(rng, idx, lang)
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"wrote {args.records} records to {args.output}")
    print("language counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
