#!/bin/bash
set -euo pipefail

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

# Usually better for multi-GPU unless you have a known P2P/NCCL issue:
# export NCCL_P2P_DISABLE=1

# Slurm typically sets CUDA_VISIBLE_DEVICES. Don't override it.
# Derive TP from what Slurm allocated:
TP="${SLURM_GPUS_ON_NODE:-1}"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --tp "$TP"
