#!/bin/bash
# Stage-1 MAE Decoder Training on RAE_DA3 Multi-Level Features
#
# Usage:
#   ./scripts/run_train_stage1_mae.sh [NUM_GPUS] [RESUME_CKPT]
#
# Examples:
#   ./scripts/run_train_stage1_mae.sh 4
#   ./scripts/run_train_stage1_mae.sh 4 results/stage1-mae/000-RAE_DA3_MAE-bf16/checkpoints/0050000.pt

NUM_GPUS="${1:-4}"
RESUME_CKPT="${2:-}"

# Set environment
export WANDB_KEY="${WANDB_KEY}"
export ENTITY="${ENTITY:-gld}"
export PROJECT="${PROJECT:-RAE_stage1}"

# PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="src:${PYTHONPATH}"

# Build resume argument
RESUME_ARG=""
if [[ -n "$RESUME_CKPT" ]]; then
    RESUME_ARG="--ckpt ${RESUME_CKPT}"
    echo "Resuming from: ${RESUME_CKPT}"
fi

echo "=========================================="
echo "  Stage-1 MAE Decoder Training"
echo "  GPUs: ${NUM_GPUS}"
echo "  Config: DA3_stage1_mae.yaml"
echo "=========================================="

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
  src/train_stage1_mae.py \
  --config configs/training/DA3_stage1_mae.yaml \
  --data-path ./ \
  --results-dir results/stage1-mae \
  --precision bf16 \
  --wandb \
  ${RESUME_ARG}
