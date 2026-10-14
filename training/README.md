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

## Evaluate A Saved Model

```bash
python training/evaluate_token_classifier.py \
  --model-dir runs/minilm-seed/model \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --jsonl data/seed/it.deepmount_coverage.seed.jsonl \
  --output runs/minilm-seed/eval.seed.json
```

## Important Modeling Rule

Records with `action=keep` are still positive NER examples when they contain a true entity. For example, `Mario Rossi` in a placeholder sentence should still train `NOME` + `COGNOME`; the risk/action layer learns that this context can be kept.
