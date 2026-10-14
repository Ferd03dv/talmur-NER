# HPC@PoliTO Cluster Notes

Source: `docs/hpcpolito_guide.pdf`.

## Access

Login node:

```bash
ssh username@hpc-legionlogin.polito.it
```

Use `scp -r` to move the repository or artifacts between local machine and cluster.

## Rules

- Do not train, compile heavy packages, or run intensive preprocessing on the login node.
- Use SLURM batch jobs via `sbatch`.
- Use interactive allocations only for debugging, and release them when done.
- Use `$HOME` for warm/persistent project files.
- Use `$SCRATCH_FLASH` for hot runtime data and outputs.
- `$SCRATCH_FLASH` files older than 30 days are automatically deleted.

## Relevant Open Partitions

CPU:

- `cpu_sapphire`: 48 cores, 512 GB RAM, 24 h walltime.
- `cpu_sapphire_ext`: 48 cores, 512 GB RAM, 120 h walltime.
- `cpu_skylake`: 32 cores, 384 GB RAM, 120 h walltime.
- `cpu_skylake_ext`: 32 cores, 384 GB RAM, 240 h walltime.

GPU:

- `gpu_a40`: 4 x A40 48 GB, 48 cores, 512 GB RAM, 24 h walltime.
- `gpu_a40_ext`: 4 x A40 48 GB, 48 cores, 512 GB RAM, 120 h walltime.
- `gpu_a100`: 4 x A100 80 GB, 48 cores, 1024 GB RAM, 24 h walltime.
- `gpu_h200`: 8 x H200 141 GB, 64 cores, 2048 GB RAM, 24 h walltime.
- `gpu_v100`: 4 x V100 32 GB, 32 cores, 384 GB RAM, 120 h walltime.
- `gpu_v100_ext`: 4 x V100 32 GB, 32 cores, 384 GB RAM, 240 h walltime.

For GPU jobs, always set `--gres=gpu:N`; submitting to a GPU partition is not enough.

## Suggested Fine-Tuning Defaults

First experiments should use one A40:

- partition: `gpu_a40`
- GPUs: `1`
- CPUs: `12`
- RAM: `96G`
- walltime: `24:00:00`

Scale memory and CPUs proportionally for more GPUs. The guide recommends roughly 12 CPU cores per A40/A100 GPU and 8 CPU cores per V100/H200 GPU.

## Workflow

1. Keep the git repo in `$HOME/talmur-NER`.
2. Prepare/generate/compile datasets with a CPU SLURM job.
3. Wait for GPU allocation only for the actual fine-tuning job.
4. The GPU job copies the prepared dataset to `$SCRATCH_FLASH/talmur-NER/$SLURM_JOB_ID`.
5. Train/evaluate in scratch.
6. Sync outputs back to `$HOME/talmur-NER/runs/...`.

Dataset generation does not require a GPU. It is text templating, surface lookup,
offset generation, validation, and JSONL compilation. Use CPU jobs for that.

## Useful Commands

Submit:

```bash
sbatch cluster/slurm/prepare_dataset_cpu.sbatch
sbatch cluster/slurm/validate_seed_cpu.sbatch
sbatch cluster/slurm/finetune_minilm_gpu.sbatch
```

Monitor:

```bash
squeue -u $USER
sacct -j <job_id>
scontrol show job <job_id>
```

Cancel:

```bash
scancel <job_id>
```

Interactive GPU debug:

```bash
salloc --partition=gpu_a40 --nodes=1 --ntasks=1 --cpus-per-task=12 --gres=gpu:1 --mem=96G --time=1:00:00
srun --pty bash
module load nvhpc-hpcx-cuda12/25.1
nvidia-smi
```
