#!/bin/bash
# Template for an external cluster: replace every EDIT and read docs/RUN_QWEN3.md first.
# One rank per GPU needs at least ~72 GB per GPU (63.4 GB of weights plus working memory). On smaller
# GPUs set GPUS_PER_RANK=2. B=8 stays global; 4 ranks take 2 prompts each.
# Resubmitting with the same OUT continues from the last checkpoint, with the same number of ranks.
#SBATCH --job-name=avrsi-a8
#SBATCH --partition=EDIT
#SBATCH --account=EDIT
#SBATCH --gres=gpu:EDIT:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/train_qwen3_%j.log
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$PWD}
PYTHON=${PYTHON:-EDIT/bin/python}
ARM=${ARM:-A8}
SEED=${SEED:-0}
STEPS=${STEPS:-1600}                # the pilot; 6400 for Phase 3
AVQA_ROOT=${AVQA_ROOT:-EDIT}        # holds train_qa.json, val_qa.json, videos/
XM_DIR=${XM_DIR:-EDIT}              # frozen train.jsonl, guard.jsonl, dev.jsonl
PLANTED=${PLANTED:-EDIT}            # guard_planted.json for this backbone
MEDIA_ROOT=${MEDIA_ROOT:-EDIT}      # muxed XM clips
OUT=${OUT:-$ROOT/outputs/qwen3/$ARM/seed$SEED}
GPUS_PER_RANK=${GPUS_PER_RANK:-1}
GPUS=$(nvidia-smi -L | wc -l)
RANKS=$((GPUS / GPUS_PER_RANK))

cd "$ROOT"
mkdir -p outputs/slurm
export PYTHONPATH=$ROOT/.venv/run_lib:$ROOT
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / RANKS)) MKL_NUM_THREADS=$((SLURM_CPUS_PER_TASK / RANKS)) TOKENIZERS_PARALLELISM=false

RESUME=()
if [ -f "$OUT/checkpoints/LATEST" ]; then RESUME=(--resume); fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$RANKS" -m avrsi.train --arm "$ARM" --model qwen3_omni --seed "$SEED" --steps "$STEPS" \
  --natural avqa --natural-root "$AVQA_ROOT" --xm-dir "$XM_DIR" --planted "$PLANTED" --media-root "$MEDIA_ROOT" --out "$OUT" \
  --gpus-per-rank "$GPUS_PER_RANK" "${RESUME[@]}"
