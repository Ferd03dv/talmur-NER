# Training

The training pipeline has separate CPU and GPU stages:

1. Validate source seed records.
2. Generate multilingual source records.
3. Merge and compile surfaces to character offsets and inspectable BIO tokens.
4. Fine-tune a Hugging Face token-classification model.

Dataset generation/compilation does not require a GPU. It should run on CPU. The GPU is only needed for model fine-tuning/evaluation.

## Compile Source Records

Validate the Italian seed without Node.js:

```bash
python training/validate_seed_dataset.py
```

```bash
python training/compile_jsonl_to_bio.py \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --input data/seed/it.deepmount_coverage.seed.jsonl \
  --output data/compiled/it.seed.compiled.jsonl
```

The compiled file is for inspection and debugging. The trainer also compiles records internally so it can keep the source JSONL as the canonical dataset.

For the preferred two-stage workflow, prepare the full seed dataset first:

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

On HPC@PoliTO:

```bash
sbatch cluster/slurm/prepare_dataset_cpu.sbatch
```

To generate a smaller debugging corpus:

```bash
SYNTHETIC_RECORDS=1000 sbatch cluster/slurm/prepare_dataset_cpu.sbatch
```

## V2: OpenPII Base And Precision Augmentation

The first corpus (`synthetic.v1.jsonl`) has template leakage and is not the
recommended base for a replacement model.  The V2 recipe starts from a
conservative mapping of OpenPII, then adds Talmur-specific labels absent from
OpenPII together with O-only hard negatives.

OpenPII publishes monolithic source files (train is roughly 4.3 GB and
validation roughly 1.1 GB). `import_openpii.py` streams them and writes only a
language-filtered snapshot; it never keeps the upstream files. The default
limit is 25,000 records per language. If a long transfer is interrupted,
rerun the exact command with `--resume` to append only unseen record ids.

```bash
python training/import_openpii.py \
  --source-split train \
  --max-per-language 25000 \
  --output data/openpii/openpii.v1.train.jsonl
```

The importer maps only unambiguous OpenPII labels. In particular, it does not
guess whether the global `TAXNUM` label is an Italian `CODICE_FISCALE` or
`P_IVA`; those must come from country-specific data.

Create the complementary precision dataset:

```bash
python training/generate_multilingual_precision_augmentation.py \
  --output data/generated/precision.multilingual.v1.jsonl
```

It contains 36,000 records: 18,000 contextual positives and 18,000 O-only hard
negatives, balanced across six languages and these 13 labels: `CVV`,
`DOSAGGIO`, `DURATION`, `FREQUENZA`, `MALATTIA`, `MEDICINA`, `ORARIO`,
`PASSWORD`, `PIN`, `PROFESSIONE`, `STATO`, `STATO_CIVILE`, and
`TARGA_VEICOLO`. The hard negatives intentionally have an empty `entities`
array. This matters: the current NER trainer treats an entity marked
`action=keep` as a positive NER span, so `keep` cannot reduce false positives.
It also includes 400 paired `MEDICINA`/`MALATTIA` clinical contrasts per
language: the positive record contains both a diagnosis and its prescribed
medicine, while the matched educational-text record labels neither. This is
intentional supervision for the medication-versus-disease boundary.

The checked-in V2 preparation uses a deterministic balanced 60,000-record
snapshot (10,000 per language) from the downloaded train checkpoint, reserving
10% for validation. This is intentionally a *development* validation set, not
a gold evaluation set:

```bash
python training/balance_openpii_snapshot.py \
  --input data/openpii/openpii.v1.train.jsonl \
  --output data/openpii/openpii.v1.balanced.jsonl \
  --per-language 10000
```

Build and inspect the V2 training source:

```bash
python training/merge_jsonl.py \
  --output data/generated/train.openpii-precision.v1.jsonl \
  data/openpii/openpii.v1.balanced.jsonl \
  data/generated/precision.multilingual.v1.jsonl

python training/compile_jsonl_to_bio.py \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --input data/generated/train.openpii-precision.v1.jsonl \
  --output data/compiled/train.openpii-precision.v1.compiled.jsonl
```

Do not train from this dataset until a held-out, human-reviewed evaluation set
is available. The OpenPII source and the augmentation are useful for a new
baseline, but both include synthetic material and are not a replacement for a
real-world gold set.

## Fine-Tune Locally Or On HPC

Minimal command:

```bash
python training/finetune_token_classifier.py \
  --model-name microsoft/Multilingual-MiniLM-L12-H384 \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --compiled-jsonl data/compiled/train.seed.all.compiled.jsonl \
  --output-dir runs/minilm-seed \
  --epochs 5 \
  --batch-size 8
```

On HPC@PoliTO:

```bash
sbatch cluster/slurm/finetune_minilm_gpu.sbatch
```

Optional overrides:

```bash
RUN_NAME=minilm-l12-v1 MODEL_NAME=microsoft/Multilingual-MiniLM-L12-H384 EPOCHS=8 BATCH_SIZE=16 \
  sbatch cluster/slurm/finetune_minilm_gpu.sbatch
```

The GPU job expects `data/compiled/train.seed.all.compiled.jsonl` to exist. Run `prepare_dataset_cpu.sbatch` before submitting the GPU job.

## Preflight Before GPU Queue

Before waiting for a GPU allocation, run the CPU preflight job:

```bash
sbatch cluster/slurm/preflight_training_cpu.sbatch
```

It checks dependencies (`protobuf`, `sentencepiece`, `torch`, `transformers`),
loads the tokenizer/config/model weights, imports `Trainer`, and validates the compiled
dataset. Only submit the GPU job after this succeeds.

Preferred guarded submission:

```bash
RUN_NAME=talos-minilm-ner-v1 TALMUR_NER_REPO=/mnt/beegfs/fdelvecchio/talmur-NER \
  bash scripts/submit_training_with_preflight.sh
```

The GPU job is submitted with an `afterok` dependency on the preflight job. If
preflight fails, GPU training does not start.

## Evaluate A Saved Model

```bash
python training/evaluate_token_classifier.py \
  --model-dir runs/minilm-seed/model \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --jsonl data/seed/it.deepmount_coverage.seed.jsonl \
  --output runs/minilm-seed/eval.seed.json
```

## Benchmark CPU Inference Latency

This benchmark forces the model onto CPU, including tokenization in its
end-to-end measurements. Use representative, held-out texts and run it on the
same CPU model that will serve production traffic.

```bash
python training/benchmark_token_classifier.py \
  --model-dir runs/minilm-seed/model \
  --jsonl data/gold/it.gold.candidate.v1.jsonl \
  --max-samples 60 \
  --warmup 20 \
  --iterations 5 \
  --threads 8 \
  --output runs/minilm-seed/benchmark.cpu.it.json
```

The output reports median, p95, p99, and mean latency in milliseconds, plus
throughput. The candidate set is useful for a smoke benchmark; use a reviewed
gold set and production-equivalent hardware for capacity decisions.

## Important Modeling Rule

Records with `action=keep` are still positive NER examples when they contain a true entity. For example, `Mario Rossi` in a placeholder sentence should still train `NOME` + `COGNOME`; the risk/action layer learns that this context can be kept.
