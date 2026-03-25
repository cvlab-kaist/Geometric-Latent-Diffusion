#!/bin/bash
# GLD Evaluation Script
#
# Usage:
#   ./eval_gld.sh                         # DA3 cascade (default)
#   ./eval_gld.sh da3 cascade             # DA3 cascade
#   ./eval_gld.sh vggt cascade            # VGGT cascade
#   ./eval_gld.sh da3 independent         # DA3 independent (level 1 only)
#   GPU_ID=1 ./eval_gld.sh               # Use GPU 1

BACKBONE="${1:-da3}"
EVAL_MODE="${2:-cascade}"
GPU_ID="${GPU_ID:-0}"

if [[ "$BACKBONE" != "da3" && "$BACKBONE" != "vggt" ]]; then
    echo "Error: backbone must be 'da3' or 'vggt'"; exit 1
fi
if [[ "$EVAL_MODE" != "independent" && "$EVAL_MODE" != "cascade" ]]; then
    echo "Error: eval_mode must be 'independent' or 'cascade'"; exit 1
fi

CONFIG_PREFIX=$(echo "$BACKBONE" | tr '[:lower:]' '[:upper:]')
if [[ "$BACKBONE" == "da3" ]]; then CONFIG_PREFIX="DA3"; fi
if [[ "$BACKBONE" == "vggt" ]]; then CONFIG_PREFIX="VGGT"; fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GLD Eval: ${BACKBONE} ${EVAL_MODE} (GPU ${GPU_ID}) ==="

CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python src/eval_gld_metric.py \
    --eval_config configs/eval/demo.yaml \
    --model_config configs/eval/${CONFIG_PREFIX}_common_eval.yaml \
    --eval_mode ${EVAL_MODE} \
    --level 1 \
    --checkpoint_level1 checkpoints/${BACKBONE}_level1.pt \
    --model_config_level1 configs/training/${CONFIG_PREFIX}_level1.yaml \
    --difficulty medium \
    --cfg_scale 1.5 \
    --output_dir results/eval_${BACKBONE}_${EVAL_MODE}"

if [[ "$EVAL_MODE" == "cascade" ]]; then
    CMD="${CMD} \
    --checkpoint_cascade checkpoints/${BACKBONE}_cascade.pt \
    --model_config_cascade configs/training/${CONFIG_PREFIX}_cascade.yaml \
    --cascade_noise_tau 0.0 \
    --cfg_scale_ar 1.5 \
    --use_camera_drop_cascade true"
fi

eval ${CMD} "${@:3}"
