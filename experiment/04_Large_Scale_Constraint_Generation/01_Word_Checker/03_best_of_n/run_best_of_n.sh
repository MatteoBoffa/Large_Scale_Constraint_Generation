##!/bin/bash

set -e
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <api-endpoint> <model>"
    exit 1
fi
# Configuration & Model
API_ENDPOINT="$1"
MODEL="$2"
N_RULES="$3"

# Base paths
REPO_ROOT="$(git rev-parse --show-toplevel)"

PATH_DATA="$REPO_ROOT/offline_datasets/processed_dataset/CommonGen/test_many_rules"
BASE_OUTPUT_PATH="$REPO_ROOT/experiment_results/WordChecker/best_of_n"
# Create base output directory
mkdir -p "$BASE_OUTPUT_PATH"

# Configuration
TEMPERATURE=0.4
TEMPLATE="Check if the following sentence contains one of the following set of words. Do not include your reasoning process in the anser; Provide a short explanation (at most 100 words) to justify your answer. Conclude your sentence with <answer>your answer</answer>, where your answer is either True or False."
N_ROUNDS=3
# Create base output directory
mkdir -p "$BASE_OUTPUT_PATH"

# Create model-specific output directory
MODEL_DIR="${BASE_OUTPUT_PATH}/${MODEL//\//_}"  # Replace / with _ in model name for safe directory naming
mkdir -p "$MODEL_DIR"
echo "Starting inference for $MODEL with $N_RULES rules"
# Construct input and output paths
INPUT_FILE="${PATH_DATA}/exact_matching_${N_RULES}.json"
OUTPUT_FILE="${MODEL_DIR}/exact_matching_${N_RULES}.json"
# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
python ../03_best_of_n/run_best_n.py \
    --input_path="$INPUT_FILE" \
    --template="$TEMPLATE" \
    --output_path="$OUTPUT_FILE" \
    --api_endpoint="$API_ENDPOINT" \
    --model_name="$MODEL" \
    --n_rounds="$N_ROUNDS" \
    --temperature="$TEMPERATURE"    
echo "Completed inference for $MODEL with $N_RULES rules"