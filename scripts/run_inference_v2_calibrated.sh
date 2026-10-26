#!/usr/bin/env bash
# Separate launcher for the fixed-calibration experiment. Extra CLI args override defaults.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
CONDA_ROOT="${CONDA_ROOT:-/home/yang/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate llf_nopo4d
VIDEO_BIN="${VIDEO_BIN:-$CONDA_ROOT/envs/llf_nopo_video/bin}"
MODEL_PYTHON="$CONDA_ROOT/envs/llf_nopo4d/bin/python"
IMAGE_DIR="${IMAGE_DIR:-assets/20261021_172146/crop_short448_view6}"
OUTPUT_DIR="${OUTPUT_DIR:-output/20261021_172146/crop_short448_view6_fixed_calibration}"
CALIBRATION="${CALIBRATION:-/home/yang/lifenglin/NoPo4D/assets/20261021_172146/extr.json}"
PATH="$VIDEO_BIN:$PATH" "$MODEL_PYTHON" src/inference_v2.py \
  --image_dir "$IMAGE_DIR" \
  --num_cameras 6 \
  --output_dir "$OUTPUT_DIR" \
  --calibration "$CALIBRATION" \
  --calibration_preprocessing config/inference/crops_20261021_172146.json \
  --calibration_profile crop_short448 \
  --calibration_translation_scale 1.0 \
  --video_frames 240 --fps 30 \
  --trajectory camera_path --camera_order 5 0 1 2 3 4 \
  --path_geometry arc --path_cycle pingpong \
  --intrinsics_mode fixed --reference_camera 0 --time_mode dynamic \
  "$@"
