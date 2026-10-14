# Taxonomy Roadmap P1-P2

Date: 2026-10-11

## P1: DeepMount Label Extraction

Source: `C:/Talos/talmur-ext/models/italian-ner-xxl-v2/config.json`.

The local DeepMount config exposes 58 base entity classes, represented as BIO labels:

- `O`: outside label.
- 58 `B-*` labels.
- 58 `I-*` labels.
- 117 model labels total.

Saved artifact:

- `taxonomy/deepmount_italian_ner_xxl_v2.labels.json`

This resolves the earlier ambiguity between 52 and 58 classes: the currently loaded local config has 58 base classes.

## P2: Talos Compatibility Mapping

Source: `C:/Talos/talmur-ext/generated-entities.js::NER_TO_CANONICAL`.

Saved artifact:

- `taxonomy/deepmount_to_talos_mapping.v1.json`

The mapping keeps every DeepMount label and maps it to the current Talos canonical entity name used by the extension/dashboard. It also adds an initial `training_role`:

- `ner`: primary responsibility should remain with token classification.
- `regex`: deterministic recognizers/validators should be primary; model examples remain useful for context and span boundaries.
- `hybrid`: model and deterministic recognizers both need to be evaluated end-to-end.

## Initial Training Implications

Classes that should be model-first:

- person/name: `NOME`, `COGNOME`
- organizations/brands: `RAGIONE_SOCIALE`, `BRAND`, `BANCA`
- locations/address semantics: `LUOGO`, `INDIRIZZO`, `STATO`
- professional/legal/clinical semantic classes: `PROFESSIONE`, `AVV_NOTAIO`, `TRIBUNALE`, `MALATTIA`, `MEDICINA`, `STORIA_CLINICA`, `REGIME_PATRIMONIALE`, `STATO_CIVILE`

Classes that should stay deterministic-first:

- `CODICE_FISCALE`, `P_IVA`, `IBAN`, `BIC`, `EMAIL`, `NUMERO_TELEFONO`, `NUMERO_CARTA`, `CVV`, `IP`, `IPV6_1`, `MAC`, `IMEI`, `TARGA_VEICOLO`, `URL`, `USER_AGENT`, `PIN`

Classes requiring hybrid treatment:

- dates/times/age: `DATA`, `DATA_NASCITA`, `DATA_MORTE`, `ORARIO`, `ETA`
- financial/legal references: `IMPORTO`, `NUMERO_CONTO`, `NUM_ASSEGNO_BANCARIO`, `N_SENTENZA`, `N_LICENZA`, `LEGGE`, `TASSO_MUTUO`, `VALUTA`
- health dosage/prescription attributes: `DOSAGGIO`, `DURATION`, `FORM`, `FREQUENZA`, `STRENGTH`
- cadastral fields: `FOGLIO`, `MAPPALE`, `PARTICELLA`, `SUBALTERNO`
- generic document identifiers: `NUMERO_DOCUMENTO`

## Open Checks Before Dataset Build

- Confirm whether Talos wants to preserve all 58 labels as training labels, or train a normalized internal taxonomy and map back to DeepMount/Talos entities.
- Decide whether retired Talos entities in the extension remain trainable labels for recall compatibility.
- Validate multilingual equivalents for Italian-specific labels, especially `CODICE_FISCALE`, `P_IVA`, cadastral labels, legal references, and health-prescription labels.
- Build a gold set where labels absent from a source dataset are not treated as `O` unless the document has been fully annotated for the whole taxonomy.
