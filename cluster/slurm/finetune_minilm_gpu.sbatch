#!/usr/bin/env bash
#SBATCH --job-name=talmur-minilm-ner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_a40
#SBATCH --mail-type=FAIL,END
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load nvhpc-hpcx-cuda12/25.1 || true

REPO_DIR="${TALMUR_NER_REPO:-$HOME/talmur-NER}"
RUN_NAME="${RUN_NAME:-minilm-l12-deepmount-it-seed}"
MODEL_NAME="${MODEL_NAME:-microsoft/Multilingual-MiniLM-L12-H384}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
COMPILED_JSONL="${COMPILED_JSONL:-data/compiled/train.seed.all.compiled.jsonl}"

if [[ -z "${SCRATCH_FLASH:-}" ]]; then
  echo "SCRATCH_FLASH is not set; this job must run on HPC@PoliTO compute nodes." >&2
  exit 1
fi

SCRATCH_RUN="$SCRATCH_FLASH/talmur-NER/$SLURM_JOB_ID"
HOME_RUN="$REPO_DIR/runs/$RUN_NAME-$SLURM_JOB_ID"

mkdir -p "$SCRATCH_RUN" "$HOME_RUN" "$REPO_DIR/logs"
rsync -a --exclude='.git' --exclude='.venv*' --exclude='runs' "$REPO_DIR/" "$SCRATCH_RUN/"

cd "$SCRATCH_RUN"

python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt

nvidia-smi || true

if [[ ! -f "$COMPILED_JSONL" ]]; then
  echo "Compiled dataset not found: $COMPILED_JSONL" >&2
  echo "Run sbatch cluster/slurm/prepare_dataset_cpu.sbatch before requesting a GPU training job." >&2
  exit 2
fi

python training/finetune_token_classifier.py \
  --model-name "$MODEL_NAME" \
  --taxonomy taxonomy/deepmount_italian_ner_xxl_v2.labels.json \
  --compiled-jsonl "$COMPILED_JSONL" \
  --output-dir "outputs/$RUN_NAME" \
  --max-length 512 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE"

rsync -a "outputs/" "$HOME_RUN/outputs/"
rsync -a "logs/" "$HOME_RUN/logs/" || true
