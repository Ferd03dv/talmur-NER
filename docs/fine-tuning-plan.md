# Fine-Tuning Plan

## Objective

Train a CPU-first multilingual token-classification student that preserves compatibility with the 58 base labels from `DeepMount00/Italian_NER_XXL_v2`, while leaving deterministic identifiers to regex/validators where they are more reliable.

## Current Step

The initial Italian seed dataset is deliberately small and exists to lock down:

- label coverage;
- record shape;
- distinction between extraction and masking policy;
- false-positive controls for names and amounts.

It is not large enough to train a production model.

## Initial Model Candidates

- `microsoft/Multilingual-MiniLM-L12-H384`
- `distilbert-base-multilingual-cased`
- optional teacher/baseline: `xlm-roberta-base`

## Training Targets

Use the DeepMount-compatible labels as the first training target so the extension mapping remains stable:

- `O`
- `B-{LABEL}` / `I-{LABEL}` for all 58 labels

Do not collapse labels until a gold-set experiment proves that normalized internal labels can be mapped back without recall loss.

## Dataset Expansion Targets

First serious Italian milestone:

- 25k-40k Italian records.
- At least 1k spans for model-first labels.
- At least 300-800 contextual examples for deterministic-first labels.
- Benign action examples for `NOME`, `COGNOME`, `IMPORTO`, `DATA`, `LUOGO`, `RAGIONE_SOCIALE`.

Production multilingual target:

- 250k-500k records total.
- Split by document/template before translation.
- Independent gold set per language.

## Evaluation

Report two families of metrics:

- NER extraction: entity F1, char recall, per-label recall, missed critical spans.
- Policy/action: false-positive masking rate, false-negative sensitive leakage rate, action accuracy.

The model can correctly detect `Ferdinando Del Vecchio` as `NOME` + `COGNOME` while the policy layer decides `keep` in benign self-introduction or placeholder contexts.
