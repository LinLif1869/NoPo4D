#!/usr/bin/env bash
# Run from any directory; resolve the project relative to this script.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ROOT="${CONDA_ROOT:-/home/yang/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate llf_nopo4d

# The video environment has a verified libx264-enabled ffmpeg.
VIDEO_BIN="$CONDA_ROOT/envs/llf_nopo_video/bin"
MODEL_PYTHON="$CONDA_ROOT/envs/llf_nopo4d/bin/python"

# Change OUTPUT_DIR for a new run: inference_v2 requires an empty directory.
IMAGE_DIR="${IMAGE_DIR:-assets/20261021_172146/crop_short448_view4}"
OUTPUT_DIR="${OUTPUT_DIR:-output/20261021_172146/crop_short448_view4_v2_retry}"

PATH="$VIDEO_BIN:$PATH" "$MODEL_PYTHON" src/inference_v2.py \
  --image_dir "$IMAGE_DIR" \
  --num_cameras 4 \
  --output_dir "$OUTPUT_DIR" \
  --video_frames 240 \
  --fps 30 \
  --trajectory camera_path \
  --path_geometry arc \
  --path_cycle pingpong \
  --intrinsics_mode fixed \
  --reference_camera 0 \
  --time_mode dynamic
