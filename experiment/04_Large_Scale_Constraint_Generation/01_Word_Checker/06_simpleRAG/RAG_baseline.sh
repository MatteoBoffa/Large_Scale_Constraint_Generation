#!/bin/bash

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
BASE_OUTPUT_PATH="$REPO_ROOT/experiment_results/WordChecker/rag_baseline"
# Create base output directory
mkdir -p "$BASE_OUTPUT_PATH"

#configuration
TEMPERATURE=0.4
TEMPLATE="Check if the following sentence contains one of the following set of words. Only answer True or False. Ensure to enclude your final answer into <answer></answer>. For instance, if the sentence contains one of the words, answer <answer>True</answer>; <answer>False</answer> otherwise."
EMBEDDER_MODEL="BAAI/bge-m3"
TOPK=20
PATH_HUGGINGFACE_CACHE="$REPO_ROOT/hf_cache"
mkdir -p "$PATH_HUGGINGFACE_CACHE"
PROMPT_PATH="../06_simpleRAG/prompts/prompt_templates.json"

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
python ../06_simpleRAG/RAG_baseline.py \
    --input_path="$INPUT_FILE" \
    --template="$TEMPLATE" \
    --output_path="$OUTPUT_FILE" \
    --api_endpoint="$API_ENDPOINT" \
    --model_name="$MODEL" \
    --temperature="$TEMPERATURE" \
    --embedder_model="$EMBEDDER_MODEL" \
    --top_k="$TOPK" \
    --hf_cache="$PATH_HUGGINGFACE_CACHE" \
    --prompt_path="$PROMPT_PATH"
echo "Completed inference for $MODEL with $N_RULES rules"
