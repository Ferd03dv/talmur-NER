#!/usr/bin/env bash
#SBATCH --job-name=talmur-prepare-dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --partition=cpu_sapphire
#SBATCH --mail-type=FAIL,END
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge

REPO_DIR="${TALMUR_NER_REPO:-$HOME/talmur-NER}"
SYNTHETIC_RECORDS="${SYNTHETIC_RECORDS:-150000}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-42}"
cd "$REPO_DIR"

mkdir -p logs data/generated data/compiled

python3 training/validate_seed_dataset.py

python3 training/generate_multilingual_seed.py \
  --output data/generated/multilingual.seed.jsonl

python3 training/generate_synthetic_corpus.py \
  --records "$SYNTHETIC_RECORDS" \
  --seed "$SYNTHETIC_SEED" \
  --output data/generated/synthetic.v1.jsonl

python3 training/merge_jsonl.py \
  --output data/generated/train.seed.all.jsonl \
  data/seed/it.deepmount_coverage.seed.jsonl \
  data/generated/multilingual.seed.jsonl \
  data/generated/synthetic.v1.jsonl

python3 training/compile_jsonl_to_bio.py \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --input data/generated/train.seed.all.jsonl \
  --output data/compiled/train.seed.all.compiled.jsonl
