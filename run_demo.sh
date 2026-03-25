#!/bin/bash
# GLD (Geometric Latent Diffusion) Demo Script
# Full pipeline: NVS inference -> 3D reconstruction (GLB + COLMAP)
#
# Prerequisites:
#   1. huggingface-cli download SeonghuJeon/GLD --local-dir .
#   2. conda activate gld
#
# Usage:
#   ./run_demo.sh da3                  # DA3 backbone, cascade pipeline
#   ./run_demo.sh vggt                 # VGGT backbone, cascade pipeline
#   ./run_demo.sh da3 0               # Specify GPU ID

BACKBONE="${1:-da3}"
GPU_ID="${2:-0}"

if [[ "$BACKBONE" != "da3" && "$BACKBONE" != "vggt" ]]; then
    echo "Error: Invalid backbone '$BACKBONE'. Must be 'da3' or 'vggt'"
    exit 1
fi

# Config paths
if [[ "$BACKBONE" == "da3" ]]; then
    CONFIG_PREFIX="DA3"
elif [[ "$BACKBONE" == "vggt" ]]; then
    CONFIG_PREFIX="VGGT"
fi

OUTPUT_DIR="results/demo_${BACKBONE}"

# Environment setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============== Step 1: Novel View Synthesis ==============
echo "=========================================="
echo "  GLD Demo (${BACKBONE} backbone)"
echo "  Step 1: Novel View Synthesis"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/eval_gld_metric.py \
    --eval_config configs/eval/demo.yaml \
    --model_config configs/eval/${CONFIG_PREFIX}_common_eval.yaml \
    --eval_mode cascade \
    --level 1 \
    --checkpoint_level1 ./checkpoints/${BACKBONE}_level1.pt \
    --model_config_level1 configs/training/${CONFIG_PREFIX}_level1.yaml \
    --checkpoint_cascade ./checkpoints/${BACKBONE}_cascade.pt \
    --model_config_cascade configs/training/${CONFIG_PREFIX}_cascade.yaml \
    --cascade_noise_tau 0.0 \
    --difficulty medium \
    --cfg_scale 1.5 \
    --output_dir ${OUTPUT_DIR}

# ============== Step 2: 3D Reconstruction ==============
echo ""
echo "=========================================="
echo "  Step 2: 3D Reconstruction"
echo "=========================================="

# Find the NPZ output from Step 1
NPZ_DIR=$(find ${OUTPUT_DIR} -name "sample_*_raw.npz" -printf '%h\n' | head -1)

if [[ -z "$NPZ_DIR" ]]; then
    echo "Warning: No NPZ files found in ${OUTPUT_DIR}. Skipping 3D reconstruction."
    echo "  (NVS step may have failed or not saved raw outputs)"
else
    echo "Found NPZ files in: ${NPZ_DIR}"
    python scripts/reconstruct_from_nvs.py \
        --input "${NPZ_DIR}" \
        --output "${OUTPUT_DIR}/reconstruction" \
        --num-max-points 500000

    echo ""
    echo "=========================================="
    echo "  Demo Complete!"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  NVS outputs:      ${OUTPUT_DIR}/"
    echo "  3D reconstruction: ${OUTPUT_DIR}/reconstruction/"
    echo "    - GLB point cloud:  reconstruction/*/scene.glb"
    echo "    - Per-view images:  reconstruction/*/views/"
    echo ""
    echo "View the GLB file in any 3D viewer (e.g. https://gltf-viewer.donmccurdy.com/)"
fi
