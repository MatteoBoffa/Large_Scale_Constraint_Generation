#!/bin/bash
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=sglang_logs/%x-%j.out
#SBATCH --mail-user=matteo.boffa@polito.it
#SBATCH --mem=64G

set -e

# Start the sglang server in the background
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
BASE_PORT=30000
N_RULES=100
#N_RULES=500
#N_RULES=1000
RANGE=20000
PORT=$((BASE_PORT + SLURM_JOB_ID % RANGE))

echo "Checking port availability..."
if ss -tulpn | grep -q ":$PORT "; then
    echo "ERROR: Port $PORT already in use on this node!"
    exit 1
fi

echo "CONFIG:"
echo "  MODEL_PATH = ${MODEL_PATH}"
echo "  PORT       = ${PORT}"
echo "  N_RULES.   = ${N_RULES}"

echo "Starting sglang server..."
MODEL_ID=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | tr -d '-')
LOG_PATH="sglang_logs/server_logs_one_shot/"
mkdir -p "$LOG_PATH"
LOG_FILE="${LOG_PATH}/sglang_server_${MODEL_ID}_nrules_${N_RULES}.log"

bash ../launch_single_gpu_sglang.sh "$MODEL_PATH" "$PORT" > "$LOG_FILE" 2>&1 &
SGLANG_PID=$!

# Give the server some time to load the model
# (you can adjust this if needed)
echo "Waiting for server to come up..."
sleep 300
echo "Timeout, moving forward!"

# Quick check that the process is still alive
if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
  echo "ERROR: sglang server process died before inference could start."
  exit 1
fi

# Run the inference script
echo "Use the model $MODEL_PATH with OneShot policy..."

API_INTERFACE="http://localhost:$PORT/v1/chat/completions"

bash ../01_simple_prompt/run_one_shot.sh "$API_INTERFACE" "$MODEL_PATH" "$N_RULES"

# After inference, stop the server
echo "Stopping sglang server..."
kill "$SGLANG_PID" 2>/dev/null || true
wait "$SGLANG_PID" 2>/dev/null || true
