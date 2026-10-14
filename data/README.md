# Dataset Layout

This repository starts with source records, not final tokenized tensors.

## Files

- `seed/it.deepmount_coverage.seed.jsonl`: Italian seed records covering the 58 DeepMount labels at least once.
- `generated/multilingual.seed.jsonl`: generated EN/FR/DE/ES/PT reference records.
- `generated/synthetic.v1.jsonl`: generated large synthetic corpus; default target is 150k records.
- `generated/train.seed.all.jsonl`: merged source records for initial multilingual experiments.
- `compiled/it.seed.compiled.jsonl`: compiled seed records with resolved character offsets and inspectable whitespace-token BIO labels.
- `compiled/train.seed.all.compiled.jsonl`: compiled merged source records for training.
- `../schemas/talmur_ner_seed_record.schema.json`: source-record schema with NER spans and sensitivity/action metadata.

## Record Semantics

Each entity has two responsibilities:

- `label`: the DeepMount-compatible NER class to train/evaluate span extraction.
- `sensitivity` and `action`: the risk-layer target used to reduce false-positive masking.

Do not train the token-classification model to ignore true names, dates, amounts, or organizations only because the action is `keep`. Those records are positive NER examples and negative/benign action examples.

## Required Compilation Step

Before fine-tuning, source records must be compiled to offset/BIO examples:

1. Locate each `surface` in `text`, using `occurrence` when the same surface appears multiple times.
2. Emit character offsets.
3. Tokenize with the selected model tokenizer.
4. Convert offsets to BIO labels.
5. Preserve `sensitivity/action/context_tags` in a sidecar file for risk-layer training and end-to-end evaluation.

Compile command:

```bash
python training/compile_jsonl_to_bio.py \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --input data/seed/it.deepmount_coverage.seed.jsonl \
  --output data/compiled/it.seed.compiled.jsonl
```

Generate and compile the initial multilingual dataset:

```bash
python training/generate_multilingual_seed.py \
  --output data/generated/multilingual.seed.jsonl

python training/generate_synthetic_corpus.py \
  --records 150000 \
  --seed 42 \
  --output data/generated/synthetic.v1.jsonl

python training/merge_jsonl.py \
  --output data/generated/train.seed.all.jsonl \
  data/seed/it.deepmount_coverage.seed.jsonl \
  data/generated/multilingual.seed.jsonl \
  data/generated/synthetic.v1.jsonl

python training/compile_jsonl_to_bio.py \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --input data/generated/train.seed.all.jsonl \
  --output data/compiled/train.seed.all.compiled.jsonl
```

On HPC@PoliTO, run the same preparation as a CPU job:

```bash
sbatch cluster/slurm/prepare_dataset_cpu.sbatch
```

## Split Rule

Splits must be done by `source.template_id` or original document family before augmentation/translation. Never put records generated from the same template into both train and test.
