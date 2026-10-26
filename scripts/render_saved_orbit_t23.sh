#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
CONDA_ROOT="${CONDA_ROOT:-/home/yang/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate llf_nopo4d
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VIDEO_BIN="${VIDEO_BIN:-$CONDA_ROOT/envs/llf_nopo_video/bin}"
SOURCE_DIR="${SOURCE_DIR:-/home/yang/lifenglin/NoPo4D/output/20261021_172146/crop_short448_view6_v2_orbit}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOURCE_DIR}_freeze_t23}"
PATH="$VIDEO_BIN:$PATH" "$CONDA_ROOT/envs/llf_nopo4d/bin/python" src/render_saved_gaussians.py \
  --source_dir "$SOURCE_DIR" --output_dir "$OUTPUT_DIR" \
  --freeze_time 2/3 --render_batch_size 1 "$@"
