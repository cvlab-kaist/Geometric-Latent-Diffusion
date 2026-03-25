#!/bin/bash
# GLD (Geometric Latent Diffusion) Training Script
#
# Usage:
#   ./run_train.sh da3 level1              # DA3 backbone, level 1
#   ./run_train.sh da3 cascade             # DA3 backbone, cascade
#   ./run_train.sh vggt level1             # VGGT backbone, level 1
#   ./run_train.sh vggt cascade            # VGGT backbone, cascade

BACKBONE="${1:?Usage: ./run_train.sh <da3|vggt> <level1|cascade>}"
MODE="${2:?Usage: ./run_train.sh <da3|vggt> <level1|cascade>}"
RUN_PREFIX="${3:-${BACKBONE}_mvdiff_cfg}"

# Resume checkpoint path (set to resume from a checkpoint)
RESUME_CKPT=""

# Validate arguments
if [[ "$BACKBONE" != "da3" && "$BACKBONE" != "vggt" ]]; then
    echo "Error: Invalid backbone '$BACKBONE'. Must be 'da3' or 'vggt'"
    exit 1
fi
if [[ "$MODE" != "level1" && "$MODE" != "cascade" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 'level1' or 'cascade'"
    exit 1
fi

# Map backbone to config prefix
if [[ "$BACKBONE" == "da3" ]]; then
    CONFIG_PREFIX="DA3"
elif [[ "$BACKBONE" == "vggt" ]]; then
    CONFIG_PREFIX="VGGT"
fi

# Build resume argument
RESUME_ARG=""
if [[ -n "$RESUME_CKPT" ]]; then
    RESUME_ARG="--ckpt ${RESUME_CKPT}"
    echo "Resuming from: ${RESUME_CKPT}"
fi

# Set environment
export WANDB_KEY="${WANDB_KEY}"  # Set your WandB API key
export ENTITY="gld"
export PROJECT="gld-${BACKBONE}"

# PyTorch memory management - reduces fragmentation for large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "  GLD Training"
echo "  Backbone: ${BACKBONE}"
echo "  Mode: ${MODE}"
echo "  Config: ${CONFIG_PREFIX}_${MODE}.yaml"
echo "=========================================="

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/train_multiview_da3.py \
  --run_name "${RUN_PREFIX}_${MODE}" \
  --config "configs/training/${CONFIG_PREFIX}_${MODE}.yaml" \
  --data-path assets \
  --results-dir "results/${BACKBONE}_mvdiffusion/${MODE}" \
  --precision bf16 \
  --wandb \
  ${RESUME_ARG}
