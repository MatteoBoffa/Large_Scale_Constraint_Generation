#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model-path> <port>"
    exit 1
fi

MODEL_PATH="$1"
PORT="$2"

module load gcc/12.4
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX="$CXX"

export NCCL_P2P_DISABLE=1
# You can override this before calling the script if needed
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=/home/mboffa/cache_models/

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --tp 1
