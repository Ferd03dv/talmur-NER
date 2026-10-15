#!/usr/bin/env bash
# Submit CPU preflight first, then GPU training only if preflight succeeds.

set -euo pipefail

REPO_DIR="${TALMUR_NER_REPO:-$(pwd)}"
RUN_NAME="${RUN_NAME:-talos-minilm-ner-v1}"

cd "$REPO_DIR"
mkdir -p logs runs

echo "repo=$REPO_DIR"
echo "run=$RUN_NAME"

PREFLIGHT_ID=$(sbatch --parsable \
  --output="$REPO_DIR/logs/talmur-preflight-%j.out" \
  --error="$REPO_DIR/logs/talmur-preflight-%j.err" \
  --export=ALL,TALMUR_NER_REPO="$REPO_DIR" \
  cluster/slurm/preflight_training_cpu.sbatch)

GPU_ID=$(sbatch --parsable \
  --dependency=afterok:"$PREFLIGHT_ID" \
  --kill-on-invalid-dep=yes \
  --output="$REPO_DIR/logs/${RUN_NAME}-%j.out" \
  --error="$REPO_DIR/logs/${RUN_NAME}-%j.err" \
  --export=ALL,TALMUR_NER_REPO="$REPO_DIR",RUN_NAME="$RUN_NAME" \
  cluster/slurm/finetune_minilm_gpu.sbatch)

echo "PREFLIGHT_ID=$PREFLIGHT_ID"
echo "GPU_ID=$GPU_ID"
echo "preflight_out=$REPO_DIR/logs/talmur-preflight-${PREFLIGHT_ID}.out"
echo "preflight_err=$REPO_DIR/logs/talmur-preflight-${PREFLIGHT_ID}.err"
echo "gpu_out=$REPO_DIR/logs/${RUN_NAME}-${GPU_ID}.out"
echo "gpu_err=$REPO_DIR/logs/${RUN_NAME}-${GPU_ID}.err"
echo "model=$REPO_DIR/runs/${RUN_NAME}-${GPU_ID}/outputs/${RUN_NAME}/model"
