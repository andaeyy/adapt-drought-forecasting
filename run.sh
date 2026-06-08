#!/bin/bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

# If you want specific conda env, activate before running.
# conda activate jupyter_env

export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# TensorFlow/GPU runtime config
: "${DROUGHTAPP_GPU_DEVICE:?Set DROUGHTAPP_GPU_DEVICE to the system GPU ID to use, for example: export DROUGHTAPP_GPU_DEVICE=0}"
export CUDA_VISIBLE_DEVICES="$DROUGHTAPP_GPU_DEVICE"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_XLA_FLAGS=--tf_xla_auto_jit=2

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
