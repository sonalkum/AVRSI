#!/bin/bash
# Two real A8 steps on local weights. Submit from the repo root: sbatch scripts/slurm/smoke_qwen3.sbatch
# The thinker is 63.4 GB at bf16 and an RTX A6000 has 48 GB, so one rank spans both GPUs by default.
# To exercise two real ranks, ask for 4 GPUs (--gres=gpu:rtxa6000:4); ranks are GPUs / GPUS_PER_RANK.
#SBATCH --job-name=avrsi-smoke
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/smoke_qwen3_%j.log
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$PWD}
PYTHON=/fs/nexus-projects/brain_project/miniconda3/envs/gama2/bin/python
GPUS_PER_RANK=${GPUS_PER_RANK:-2}
GPUS=$(nvidia-smi -L | wc -l)
RANKS=$((GPUS / GPUS_PER_RANK))
OUT=${OUT:-$ROOT/outputs/smoke_qwen3/$SLURM_JOB_ID}

cd "$ROOT"
mkdir -p outputs/slurm "$OUT"
export PYTHONPATH=$ROOT/.venv/run_lib:$ROOT
export HF_HUB_CACHE=/fs/gamma-projects/audio/.cache/hub HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / RANKS)) MKL_NUM_THREADS=$((SLURM_CPUS_PER_TASK / RANKS)) TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$RANKS" scripts/smoke_qwen3.py --gpus-per-rank "$GPUS_PER_RANK" --out "$OUT" "$@"
