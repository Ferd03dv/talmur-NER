#!/usr/bin/env bash
#SBATCH --job-name=talmur-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --partition=cpu_sapphire
#SBATCH --mail-type=FAIL,END
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge

REPO_DIR="${TALMUR_NER_REPO:-$HOME/talmur-NER}"
MODEL_NAME="${MODEL_NAME:-microsoft/Multilingual-MiniLM-L12-H384}"
COMPILED_JSONL="${COMPILED_JSONL:-data/compiled/train.seed.all.compiled.jsonl}"

cd "$REPO_DIR"
mkdir -p logs .cache/huggingface

export HF_HOME="$REPO_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=false

python3 -m venv .venv-preflight
source .venv-preflight/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python -m pip check

python training/preflight_training_env.py \
  --model-name "$MODEL_NAME" \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --compiled-jsonl "$COMPILED_JSONL" \
  --min-records 1000
