#!/usr/bin/env python3
"""Generate multilingual, context-sensitive data for labels absent from OpenPII.

Every label/language pair receives both positive examples and *true O-only hard
negatives*.  The latter mention similar words, units, or document contexts but
do not contain a value that should be redacted.  They are essential for
reducing false positives; ``action=keep`` is not enough because the current
token-classification trainer still learns it as a positive entity.
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
    "PROFESSIONE": "PROFESSION", "STATO": "COUNTRY", "STATO_CIVILE": "MARITAL_STATUS",
    "TARGA_VEICOLO": "VEHICLE_PLATE",
}

# Each string has exactly one ``{value}`` marker.  These are intentionally
# natural document fragments rather than label-name prompts.
POSITIVE = {
    "it": {
        "CVV": ["Per autorizzare il rimborso, il codice di sicurezza della carta è {value}.", "Nel modulo di pagamento è stato indicato il CVV {value}."],
        "DOSAGGIO": ["Prescrizione: assumere {value} di medicinale dopo i pasti.", "La terapia registrata prevede una dose di {value}."],
        "DURATION": ["Il trattamento deve proseguire per {value}.", "La prescrizione è valida per una durata di {value}."],
        "FREQUENZA": ["Assumere il farmaco {value}.", "La frequenza prevista della terapia è {value}."],
        "MALATTIA": ["La cartella del paziente riporta diagnosi di {value}.", "È in cura per {value} presso l'ambulatorio."],
        "MEDICINA": ["Il medico ha prescritto {value}.", "La terapia farmacologica include {value}."],
        "ORARIO": ["L'appuntamento sanitario è fissato alle {value}.", "Il prelievo del paziente è stato registrato alle {value}."],
        "PASSWORD": ["La password temporanea dell'account è {value}.", "Per accedere al fascicolo usare la password {value}."],
        "PIN": ["Il PIN personale consegnato al cliente è {value}.", "Per confermare l'operazione inserire il PIN {value}."],
        "PROFESSIONE": ["Nella scheda personale la professione indicata è {value}.", "Il candidato lavora come {value}."],
        "STATO": ["Il paese di residenza dichiarato dal cliente è {value}.", "Nella pratica anagrafica risulta cittadinanza {value}."],
        "STATO_CIVILE": ["Nella dichiarazione anagrafica lo stato civile è {value}.", "Il profilo personale riporta stato civile {value}."],
        "TARGA_VEICOLO": ["Il veicolo associato al cliente ha targa {value}.", "Nel sinistro è stata registrata la targa {value}."],
    },
    "en": {
        "CVV": ["To authorise the refund, the card security code is {value}.", "The payment form contains CVV {value}."],
        "DOSAGGIO": ["Prescription: take {value} of the medicine after meals.", "The recorded treatment requires a dose of {value}."],
        "DURATION": ["The treatment must continue for {value}.", "The prescription has a duration of {value}."],
        "FREQUENZA": ["Take the medicine {value}.", "The prescribed treatment frequency is {value}."],
        "MALATTIA": ["The patient's record lists a diagnosis of {value}.", "The patient is receiving care for {value}."],
        "MEDICINA": ["The doctor prescribed {value}.", "The medication plan includes {value}."],
        "ORARIO": ["The medical appointment is scheduled at {value}.", "The patient's sample was recorded at {value}."],
        "PASSWORD": ["The temporary account password is {value}.", "Use password {value} to access the case file."],
        "PIN": ["The personal PIN issued to the customer is {value}.", "Enter PIN {value} to confirm the operation."],
        "PROFESSIONE": ["The personal record lists the occupation as {value}.", "The applicant works as a {value}."],
        "STATO": ["The customer's declared country of residence is {value}.", "The personal record lists nationality {value}."],
        "STATO_CIVILE": ["The civil-status declaration says {value}.", "The personal profile lists marital status as {value}."],
        "TARGA_VEICOLO": ["The vehicle assigned to the customer has licence plate {value}.", "The accident report records plate {value}."],
    },
    "fr": {
        "CVV": ["Pour autoriser le remboursement, le code de sécurité de la carte est {value}.", "Le formulaire de paiement indique le CVV {value}."],
        "DOSAGGIO": ["Ordonnance : prendre {value} de médicament après les repas.", "Le traitement enregistré prévoit une dose de {value}."],
        "DURATION": ["Le traitement doit durer {value}.", "La prescription est prévue pour une durée de {value}."],
        "FREQUENZA": ["Prendre le médicament {value}.", "La fréquence du traitement prescrite est {value}."],
        "MALATTIA": ["Le dossier du patient mentionne un diagnostic de {value}.", "Le patient est suivi pour {value}."],
        "MEDICINA": ["Le médecin a prescrit {value}.", "Le plan de traitement comprend {value}."],
        "ORARIO": ["Le rendez-vous médical est fixé à {value}.", "Le prélèvement du patient a été enregistré à {value}."],
        "PASSWORD": ["Le mot de passe temporaire du compte est {value}.", "Utilisez le mot de passe {value} pour accéder au dossier."],
        "PIN": ["Le code PIN personnel remis au client est {value}.", "Saisissez le PIN {value} pour confirmer l'opération."],
        "PROFESSIONE": ["La fiche personnelle indique la profession {value}.", "Le candidat travaille comme {value}."],
        "STATO": ["Le pays de résidence déclaré par le client est {value}.", "Le dossier personnel indique la nationalité {value}."],
        "STATO_CIVILE": ["La déclaration d'état civil indique {value}.", "Le profil personnel indique la situation matrimoniale {value}."],
        "TARGA_VEICOLO": ["Le véhicule associé au client porte la plaque {value}.", "Le constat d'accident enregistre la plaque {value}."],
    },
    "de": {
        "CVV": ["Zur Freigabe der Erstattung lautet der Sicherheitscode der Karte {value}.", "Im Zahlungsformular steht der CVV {value}."],
        "DOSAGGIO": ["Verordnung: Nach dem Essen {value} des Medikaments einnehmen.", "Die dokumentierte Therapie sieht eine Dosis von {value} vor."],
        "DURATION": ["Die Behandlung soll {value} dauern.", "Die Verordnung gilt für eine Dauer von {value}."],
        "FREQUENZA": ["Das Medikament {value} einnehmen.", "Die verordnete Therapiefrequenz lautet {value}."],
        "MALATTIA": ["In der Patientenakte steht die Diagnose {value}.", "Die Person wird wegen {value} behandelt."],
        "MEDICINA": ["Der Arzt hat {value} verschrieben.", "Der Medikationsplan enthält {value}."],
        "ORARIO": ["Der medizinische Termin ist um {value} Uhr.", "Die Probe des Patienten wurde um {value} Uhr erfasst."],
        "PASSWORD": ["Das temporäre Passwort des Kontos lautet {value}.", "Für die Akte das Passwort {value} verwenden."],
        "PIN": ["Die persönliche PIN des Kunden ist {value}.", "Zur Bestätigung die PIN {value} eingeben."],
        "PROFESSIONE": ["In der Personalakte ist als Beruf {value} angegeben.", "Der Bewerber arbeitet als {value}."],
        "STATO": ["Das angegebene Wohnsitzland des Kunden ist {value}.", "Die Personenakte nennt die Staatsangehörigkeit {value}."],
        "STATO_CIVILE": ["Die Personenstandserklärung nennt {value}.", "Im persönlichen Profil steht Familienstand {value}."],
        "TARGA_VEICOLO": ["Das Fahrzeug des Kunden hat das Kennzeichen {value}.", "Im Unfallbericht ist das Kennzeichen {value} vermerkt."],
    },
    "es": {
        "CVV": ["Para autorizar el reembolso, el código de seguridad de la tarjeta es {value}.", "El formulario de pago contiene el CVV {value}."],
        "DOSAGGIO": ["Receta: tomar {value} del medicamento después de las comidas.", "El tratamiento registrado requiere una dosis de {value}."],
        "DURATION": ["El tratamiento debe continuar durante {value}.", "La receta tiene una duración de {value}."],
        "FREQUENZA": ["Tomar el medicamento {value}.", "La frecuencia prescrita del tratamiento es {value}."],
        "MALATTIA": ["La historia del paciente indica un diagnóstico de {value}.", "El paciente recibe atención por {value}."],
        "MEDICINA": ["El médico recetó {value}.", "El plan farmacológico incluye {value}."],
        "ORARIO": ["La cita médica está programada a las {value}.", "La muestra del paciente se registró a las {value}."],
        "PASSWORD": ["La contraseña temporal de la cuenta es {value}.", "Use la contraseña {value} para acceder al expediente."],
        "PIN": ["El PIN personal entregado al cliente es {value}.", "Introduzca el PIN {value} para confirmar la operación."],
        "PROFESSIONE": ["La ficha personal indica la profesión {value}.", "La persona trabaja como {value}."],
        "STATO": ["El país de residencia declarado por el cliente es {value}.", "El registro personal indica nacionalidad {value}."],
        "STATO_CIVILE": ["La declaración de estado civil indica {value}.", "El perfil personal muestra estado civil {value}."],
        "TARGA_VEICOLO": ["El vehículo asociado al cliente tiene matrícula {value}.", "El informe del accidente registra la matrícula {value}."],
    },
    "pt": {
        "CVV": ["Para autorizar o reembolso, o código de segurança do cartão é {value}.", "O formulário de pagamento contém o CVV {value}."],
        "DOSAGGIO": ["Prescrição: tomar {value} do medicamento após as refeições.", "O tratamento registado requer uma dose de {value}."],
        "DURATION": ["O tratamento deve continuar por {value}.", "A receita tem uma duração de {value}."],
        "FREQUENZA": ["Tomar o medicamento {value}.", "A frequência prescrita do tratamento é {value}."],
        "MALATTIA": ["O processo do paciente indica diagnóstico de {value}.", "O paciente recebe cuidados por {value}."],
        "MEDICINA": ["O médico prescreveu {value}.", "O plano de medicação inclui {value}."],
        "ORARIO": ["A consulta médica está marcada para as {value}.", "A amostra do paciente foi registada às {value}."],
        "PASSWORD": ["A palavra-passe temporária da conta é {value}.", "Use a palavra-passe {value} para aceder ao processo."],
        "PIN": ["O PIN pessoal entregue ao cliente é {value}.", "Introduza o PIN {value} para confirmar a operação."],
        "PROFESSIONE": ["A ficha pessoal indica a profissão {value}.", "A pessoa trabalha como {value}."],
        "STATO": ["O país de residência declarado pelo cliente é {value}.", "O processo pessoal indica nacionalidade {value}."],
        "STATO_CIVILE": ["A declaração de estado civil indica {value}.", "O perfil pessoal mostra estado civil {value}."],
        "TARGA_VEICOLO": ["O veículo associado ao cliente tem matrícula {value}.", "O relatório do acidente regista a matrícula {value}."],
    },
}

VALUES = {
    "CVV": lambda rng, lang: f"{rng.randrange(0, 1000):03d}",
    "DOSAGGIO": lambda rng, lang: f"{rng.choice((125, 250, 400, 500, 750, 1000))} mg",
    "DURATION": lambda rng, lang: {"it": f"{rng.randint(3, 90)} giorni", "en": f"{rng.randint(3, 90)} days", "fr": f"{rng.randint(3, 90)} jours", "de": f"{rng.randint(3, 90)} Tage", "es": f"{rng.randint(3, 90)} días", "pt": f"{rng.randint(3, 90)} dias"}[lang],
    "FREQUENZA": lambda rng, lang: {"it": rng.choice(("una volta al giorno", "due volte al giorno", "ogni 8 ore")), "en": rng.choice(("once a day", "twice a day", "every 8 hours")), "fr": rng.choice(("une fois par jour", "deux fois par jour", "toutes les 8 heures")), "de": rng.choice(("einmal täglich", "zweimal täglich", "alle 8 Stunden")), "es": rng.choice(("una vez al día", "dos veces al día", "cada 8 horas")), "pt": rng.choice(("uma vez por dia", "duas vezes por dia", "a cada 8 horas"))}[lang],
    "MALATTIA": lambda rng, lang: {"it": rng.choice(("ipertensione", "diabete", "asma", "emicrania")), "en": rng.choice(("hypertension", "diabetes", "asthma", "migraine")), "fr": rng.choice(("hypertension", "diabète", "asthme", "migraine")), "de": rng.choice(("Bluthochdruck", "Diabetes", "Asthma", "Migräne")), "es": rng.choice(("hipertensión", "diabetes", "asma", "migraña")), "pt": rng.choice(("hipertensão", "diabetes", "asma", "enxaqueca"))}[lang],
    "MEDICINA": lambda rng, lang: rng.choice(("Metformina", "Amoxicillina", "Ibuprofene", "Atorvastatina")) if lang == "it" else rng.choice(("Metformin", "Amoxicillin", "Ibuprofen", "Atorvastatin")),
    "ORARIO": lambda rng, lang: f"{rng.randrange(7, 20):02d}:{rng.randrange(0, 60):02d}",
    "PASSWORD": lambda rng, lang: f"{rng.choice(('Luna', 'Rosa', 'Atlas', 'Mare'))}!{rng.randrange(1000, 9999)}",
    "PIN": lambda rng, lang: f"{rng.randrange(0, 10000):04d}",
    "PROFESSIONE": lambda rng, lang: {"it": rng.choice(("medico", "avvocato", "ingegnere", "contabile")), "en": rng.choice(("doctor", "lawyer", "engineer", "accountant")), "fr": rng.choice(("médecin", "avocat", "ingénieur", "comptable")), "de": rng.choice(("Arzt", "Anwalt", "Ingenieur", "Buchhalter")), "es": rng.choice(("médico", "abogado", "ingeniero", "contable")), "pt": rng.choice(("médico", "advogado", "engenheiro", "contabilista"))}[lang],
    "STATO": lambda rng, lang: {"it": rng.choice(("Italia", "Francia", "Germania", "Spagna")), "en": rng.choice(("Italy", "France", "Germany", "Spain")), "fr": rng.choice(("Italie", "France", "Allemagne", "Espagne")), "de": rng.choice(("Italien", "Frankreich", "Deutschland", "Spanien")), "es": rng.choice(("Italia", "Francia", "Alemania", "España")), "pt": rng.choice(("Itália", "França", "Alemanha", "Espanha"))}[lang],
    "STATO_CIVILE": lambda rng, lang: {"it": rng.choice(("coniugato", "nubile", "divorziato", "vedovo")), "en": rng.choice(("married", "single", "divorced", "widowed")), "fr": rng.choice(("marié", "célibataire", "divorcé", "veuf")), "de": rng.choice(("verheiratet", "ledig", "geschieden", "verwitwet")), "es": rng.choice(("casado", "soltero", "divorciado", "viudo")), "pt": rng.choice(("casado", "solteiro", "divorciado", "viúvo"))}[lang],
    "TARGA_VEICOLO": lambda rng, lang: {"it": f"{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.randrange(100, 1000)}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}", "en": f"{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.randrange(10, 100)} {rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}", "fr": f"{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}-{rng.randrange(100, 1000)}-{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}", "de": f"B-{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')} {rng.randrange(1000, 10000)}", "es": f"{rng.randrange(1000, 10000)} {rng.choice('BCDFGHJKLMNPRSTVWXYZ')}{rng.choice('BCDFGHJKLMNPRSTVWXYZ')}{rng.choice('BCDFGHJKLMNPRSTVWXYZ')}", "pt": f"{rng.randrange(10, 100)}-{rng.randrange(10, 100)}-{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXYZ')}"}[lang],
}

NEGATIVE = {
    "it": {"CVV": "Il manuale spiega che CVV è una sigla dei pagamenti, senza riportare codici.", "DOSAGGIO": "La scatola pesa 500 g e non contiene indicazioni terapeutiche.", "DURATION": "Il progetto editoriale dura trenta giorni, senza dati sanitari.", "FREQUENZA": "La riunione del team si ripete ogni otto ore durante il test.", "MALATTIA": "Il corso universitario discute l'ipertensione come tema di salute pubblica.", "MEDICINA": "Il titolo dell'articolo cita Metformina in una bibliografia scientifica.", "ORARIO": "Il treno parte alle 08:30 secondo l'orario pubblico.", "PASSWORD": "La guida descrive il termine password senza fornire credenziali.", "PIN": "La tastiera ha un tasto PIN, ma nessun codice personale.", "PROFESSIONE": "Il testo descrive la professione medica in generale.", "STATO": "La Francia è membro dell'Unione europea.", "STATO_CIVILE": "Il questionario statistico contiene la voce stato civile senza dati personali.", "TARGA_VEICOLO": "Il regolamento spiega dove trovare la targa di un veicolo."},
    "en": {"CVV": "The guide explains that CVV is a payment acronym without listing any code.", "DOSAGGIO": "The box weighs 500 g and contains no treatment instruction.", "DURATION": "The editorial project lasts thirty days and contains no health data.", "FREQUENZA": "The team meeting repeats every eight hours during the test.", "MALATTIA": "The university course discusses hypertension as a public-health topic.", "MEDICINA": "The article title cites Metformin in a scientific bibliography.", "ORARIO": "The train leaves at 08:30 according to the public timetable.", "PASSWORD": "The guide defines the word password without providing credentials.", "PIN": "The keypad has a PIN key but no personal code.", "PROFESSIONE": "The text describes the medical profession in general.", "STATO": "France is a member of the European Union.", "STATO_CIVILE": "The statistical questionnaire has a marital-status field without personal data.", "TARGA_VEICOLO": "The regulation explains where to find a vehicle licence plate."},
    "fr": {"CVV": "Le guide explique que CVV est un sigle de paiement sans donner de code.", "DOSAGGIO": "La boîte pèse 500 g et ne contient aucune consigne thérapeutique.", "DURATION": "Le projet éditorial dure trente jours sans données de santé.", "FREQUENZA": "La réunion d'équipe se répète toutes les huit heures pendant le test.", "MALATTIA": "Le cours universitaire aborde l'hypertension comme sujet de santé publique.", "MEDICINA": "Le titre de l'article cite Metformin dans une bibliographie scientifique.", "ORARIO": "Le train part à 08:30 selon l'horaire public.", "PASSWORD": "Le guide définit le mot mot de passe sans fournir d'identifiant.", "PIN": "Le clavier possède une touche PIN mais aucun code personnel.", "PROFESSIONE": "Le texte décrit la profession médicale en général.", "STATO": "La France est membre de l'Union européenne.", "STATO_CIVILE": "Le questionnaire statistique contient un champ d'état civil sans donnée personnelle.", "TARGA_VEICOLO": "Le règlement explique où trouver la plaque d'un véhicule."},
    "de": {"CVV": "Der Leitfaden erklärt, dass CVV eine Zahlungsabkürzung ist, ohne einen Code zu nennen.", "DOSAGGIO": "Die Schachtel wiegt 500 g und enthält keine Therapieanweisung.", "DURATION": "Das Redaktionsprojekt dauert dreißig Tage und enthält keine Gesundheitsdaten.", "FREQUENZA": "Das Teamtreffen wiederholt sich im Test alle acht Stunden.", "MALATTIA": "Der Universitätskurs behandelt Bluthochdruck als Public-Health-Thema.", "MEDICINA": "Der Artikeltitel nennt Metformin in einer wissenschaftlichen Bibliografie.", "ORARIO": "Der Zug fährt laut öffentlichem Fahrplan um 08:30 Uhr ab.", "PASSWORD": "Der Leitfaden erklärt das Wort Passwort ohne Zugangsdaten.", "PIN": "Die Tastatur hat eine PIN-Taste, aber keinen persönlichen Code.", "PROFESSIONE": "Der Text beschreibt den Arztberuf allgemein.", "STATO": "Frankreich ist Mitglied der Europäischen Union.", "STATO_CIVILE": "Der Statistikfragebogen enthält ein Feld zum Familienstand ohne Personendaten.", "TARGA_VEICOLO": "Die Regelung erklärt, wo sich ein Fahrzeugkennzeichen befindet."},
    "es": {"CVV": "La guía explica que CVV es una sigla de pago sin indicar códigos.", "DOSAGGIO": "La caja pesa 500 g y no contiene instrucciones terapéuticas.", "DURATION": "El proyecto editorial dura treinta días sin datos sanitarios.", "FREQUENZA": "La reunión del equipo se repite cada ocho horas durante la prueba.", "MALATTIA": "El curso universitario trata la hipertensión como tema de salud pública.", "MEDICINA": "El título del artículo cita Metformin en una bibliografía científica.", "ORARIO": "El tren sale a las 08:30 según el horario público.", "PASSWORD": "La guía define la palabra contraseña sin proporcionar credenciales.", "PIN": "El teclado tiene una tecla PIN pero ningún código personal.", "PROFESSIONE": "El texto describe la profesión médica en general.", "STATO": "Francia es miembro de la Unión Europea.", "STATO_CIVILE": "El cuestionario estadístico contiene un campo de estado civil sin datos personales.", "TARGA_VEICOLO": "El reglamento explica dónde encontrar la matrícula de un vehículo."},
    "pt": {"CVV": "O guia explica que CVV é uma sigla de pagamento sem indicar códigos.", "DOSAGGIO": "A caixa pesa 500 g e não contém instruções terapêuticas.", "DURATION": "O projeto editorial dura trinta dias sem dados de saúde.", "FREQUENZA": "A reunião da equipa repete-se a cada oito horas durante o teste.", "MALATTIA": "O curso universitário aborda a hipertensão como tema de saúde pública.", "MEDICINA": "O título do artigo cita Metformin numa bibliografia científica.", "ORARIO": "O comboio parte às 08:30 segundo o horário público.", "PASSWORD": "O guia define a palavra palavra-passe sem fornecer credenciais.", "PIN": "O teclado tem uma tecla PIN, mas nenhum código pessoal.", "PROFESSIONE": "O texto descreve a profissão médica em geral.", "STATO": "A França é membro da União Europeia.", "STATO_CIVILE": "O questionário estatístico contém um campo de estado civil sem dados pessoais.", "TARGA_VEICOLO": "O regulamento explica onde encontrar a matrícula de um veículo."},
}

# A token classifier learns the MEDICINA/MALATTIA boundary far more reliably
# when both concepts occur in the *same* clinical context.  The paired
# negatives retain the words but remove the patient-specific context, teaching
# the intended masking policy rather than a simplistic lexical association.
CLINICAL_CONTRAST = {
    "it": (
        "Diagnosi del paziente: {disease}. Terapia prescritta: {medicine}.",
        "L'articolo confronta {disease} e {medicine} in una rassegna scientifica generale.",
    ),
    "en": (
        "Patient diagnosis: {disease}. Prescribed medication: {medicine}.",
        "The article compares {disease} and {medicine} in a general scientific review.",
    ),
    "fr": (
        "Diagnostic du patient : {disease}. Médicament prescrit : {medicine}.",
        "L'article compare {disease} et {medicine} dans une revue scientifique générale.",
    ),
    "de": (
        "Patientendiagnose: {disease}. Verordnetes Medikament: {medicine}.",
        "Der Artikel vergleicht {disease} und {medicine} in einer allgemeinen wissenschaftlichen Übersicht.",
    ),
    "es": (
        "Diagnóstico del paciente: {disease}. Medicamento recetado: {medicine}.",
        "El artículo compara {disease} y {medicine} en una revisión científica general.",
    ),
    "pt": (
        "Diagnóstico do paciente: {disease}. Medicamento prescrito: {medicine}.",
        "O artigo compara {disease} e {medicine} numa revisão científica geral.",
    ),
}


def split_for(index: int, validation_ratio: float) -> str:
    return "validation" if index % round(1 / validation_ratio) == 0 else "train"


def positive_record(lang: str, label: str, value: str, index: int, rng: random.Random, validation_ratio: float) -> dict:
    text = rng.choice(POSITIVE[lang][label]).format(value=value)
    return {
        "id": f"precision-{lang}-{label.lower()}-positive-{index:06d}", "language": lang,
        "split": split_for(index, validation_ratio),
        "source": {"kind": "synthetic_generated", "template_id": f"precision_positive_{label.lower()}_v1", "document_type": "contextual_sensitive_record", "notes": "Multilingual augmentation for false-positive reduction."},
        "text": text,
        "entities": [{"surface": value, "label": label, "canonical_entity": CANONICAL[label], "sensitivity": "critical" if label in {"CVV", "PASSWORD", "PIN"} else "high", "action": "mask", "context_tags": ["precision_augmentation", "positive_context"]}],
        "record_tags": ["precision_augmentation_v1", "positive", lang, label],
    }


def negative_record(lang: str, label: str, index: int, validation_ratio: float) -> dict:
    # The changing reference makes this a distinct document without adding a
    # phrase that looks like the entity itself.
    reference = {"it": "Riferimento interno", "en": "Internal reference", "fr": "Référence interne", "de": "Interne Referenz", "es": "Referencia interna", "pt": "Referência interna"}[lang]
    text = f"{NEGATIVE[lang][label]} {reference} {index:05d}."
    return {
        "id": f"precision-{lang}-{label.lower()}-negative-{index:06d}", "language": lang,
        "split": split_for(index, validation_ratio),
        "source": {"kind": "synthetic_generated", "template_id": f"precision_negative_{label.lower()}_v1", "document_type": "benign_context", "notes": "O-only hard negative; no span should be redacted."},
        "text": text, "entities": [],
        "record_tags": ["precision_augmentation_v1", "hard_negative", lang, label],
    }


def clinical_contrast_records(lang: str, index: int, rng: random.Random, validation_ratio: float) -> tuple[dict, dict]:
    disease = VALUES["MALATTIA"](rng, lang)
    medicine = VALUES["MEDICINA"](rng, lang)
    positive_template, negative_template = CLINICAL_CONTRAST[lang]
    positive_text = positive_template.format(disease=disease, medicine=medicine)
    negative_text = negative_template.format(disease=disease, medicine=medicine)
    split = split_for(index, validation_ratio)
    positive = {
        "id": f"precision-{lang}-clinical-contrast-positive-{index:06d}", "language": lang, "split": split,
        "source": {"kind": "synthetic_generated", "template_id": "clinical_contrast_positive_v1", "document_type": "clinical_record", "notes": "Paired MEDICINA/MALATTIA supervision."},
        "text": positive_text,
        "entities": [
            {"surface": disease, "label": "MALATTIA", "canonical_entity": CANONICAL["MALATTIA"], "sensitivity": "high", "action": "mask", "context_tags": ["clinical_contrast", "diagnosis"]},
            {"surface": medicine, "label": "MEDICINA", "canonical_entity": CANONICAL["MEDICINA"], "sensitivity": "high", "action": "mask", "context_tags": ["clinical_contrast", "prescription"]},
        ],
        "record_tags": ["precision_augmentation_v1", "clinical_contrast", "positive", lang],
    }
    negative = {
        "id": f"precision-{lang}-clinical-contrast-negative-{index:06d}", "language": lang, "split": split,
        "source": {"kind": "synthetic_generated", "template_id": "clinical_contrast_negative_v1", "document_type": "medical_education", "notes": "O-only contrast: not a patient record and not redacted."},
        "text": negative_text, "entities": [],
        "record_tags": ["precision_augmentation_v1", "clinical_contrast", "hard_negative", lang],
    }
    return positive, negative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", default="it,en,fr,de,es,pt")
    parser.add_argument("--positive-per-label", type=int, default=200)
    parser.add_argument("--negative-per-label", type=int, default=200)
    parser.add_argument("--clinical-contrast-pairs", type=int, default=400)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if min(args.positive_per_label, args.negative_per_label, args.clinical_contrast_pairs) < 1 or not 0 < args.validation_ratio < 1:
        raise SystemExit("example counts must be positive and validation ratio must be between 0 and 1")
    languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
    unknown = set(languages) - set(POSITIVE)
    if unknown:
        raise SystemExit("unsupported languages: " + ", ".join(sorted(unknown)))

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as out:
        for lang in languages:
            for label in TARGET_LABELS:
                for i in range(1, args.positive_per_label + 1):
                    record = positive_record(lang, label, VALUES[label](rng, lang), i, rng, args.validation_ratio)
                    out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
                for i in range(1, args.negative_per_label + 1):
                    record = negative_record(lang, label, i, args.validation_ratio)
                    out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
            for i in range(1, args.clinical_contrast_pairs + 1):
                positive, negative = clinical_contrast_records(lang, i, rng, args.validation_ratio)
                out.write(json.dumps(positive, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.write(json.dumps(negative, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 2
    print(f"wrote {count} records to {args.output}")
    per_language = len(TARGET_LABELS) * (args.positive_per_label + args.negative_per_label) + (2 * args.clinical_contrast_pairs)
    print(f"per language: {per_language}; labels: {len(TARGET_LABELS)}; clinical contrast pairs: {args.clinical_contrast_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
