#!/bin/bash
# GLD 3D Reconstruction from NVS Results
#
# Takes NVS output (pred_rgb + cameras) and:
#   1. Runs DA3 depth estimation on predicted RGB images
#   2. Reconstructs 3D point cloud (GLB + COLMAP)
#
# Usage:
#   ./scripts/run_3d_recon.sh [NVS_OUTPUT_DIR] [GPU_ID]
#
# Examples:
#   ./scripts/run_3d_recon.sh results/demo_da3                # Reconstruct from NVS results
#   ./scripts/run_3d_recon.sh results/demo_da3 0              # Use GPU 0

NVS_DIR="${1:?Usage: $0 <nvs_output_dir> [gpu_id]}"
GPU_ID="${2:-0}"

# Find NPZ files
NPZ_DIR=$(find "${NVS_DIR}" -name "sample_*_raw.npz" -printf '%h\n' | head -1)

if [[ -z "$NPZ_DIR" ]]; then
    echo "Error: No sample_*_raw.npz files found in ${NVS_DIR}"
    exit 1
fi

echo "=========================================="
echo "  GLD 3D Reconstruction"
echo "  Input: ${NPZ_DIR}"
echo "  GPU: ${GPU_ID}"
echo "=========================================="

OUTPUT_DIR="${NVS_DIR}/reconstruction"

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/reconstruct_npz_scene.py \
    --input "${NPZ_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --camera-source pred \
    --sim3 \
    --use-pred-ray-conf \
    --export glb \
    --num-max-points 500000

echo ""
echo "Results:"
echo "  3D reconstruction: ${OUTPUT_DIR}/"
echo "  View GLB at: https://gltf-viewer.donmccurdy.com/"
