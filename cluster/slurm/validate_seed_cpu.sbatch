#!/usr/bin/env bash
#SBATCH --job-name=talmur-seed-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --partition=cpu_sapphire
#SBATCH --mail-type=FAIL,END
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge

REPO_DIR="${TALMUR_NER_REPO:-$HOME/talmur-NER}"
cd "$REPO_DIR"

mkdir -p logs
python3 training/validate_seed_dataset.py
