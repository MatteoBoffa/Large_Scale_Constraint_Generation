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

PY="python" 
REPO_ROOT="$(git rev-parse --show-toplevel)"

#GPU_ID
GPU_ID=0
# Base paths
PATH_DATA="$REPO_ROOT/offline_datasets/processed_dataset/CommonGen/test_many_rules"
BASE_OUTPUT_PATH="$REPO_ROOT/experiment_results/WordChecker/FocusNet"
# Details about validator
PROJECT="rule_constrainer"
GROUP="common_net"
LOSS_RULES="True"
H_OUT=128
H_IN=768
TEMPERATURE_RULE_CHECKER=0.05
CLASSIFIER="knn"
NORMALIZE_EMBEDDINGS="True"
EMBEDDINGS_AGGR="mean_pooling"
SENTENCE_ENCODER="sentence-transformers/all-mpnet-base-v2"
PATH_PROMPT_TEMPLATE="../04_FocusNet/prompts/prompt_templates.json"
PATH_HUGGINGFACE_CACHE="$REPO_ROOT/hf_cache"
EXPERIMENT_ID="knn_training_hout_${H_OUT}_loss_rules_${LOSS_RULES}"
VALIDATOR_PATH="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID/training_best_model/best_model"

TEMPERATURE_LLM=0.2
TEMPLATE="Check if the following sentence contains one of the following set of words. Only answer True or False. Ensure to enclude your final answer into <answer></answer>. For instance, if the sentence contains one of the words, answer <answer>True</answer>; <answer>False</answer> otherwise."
# Create base output directory
mkdir -p "$BASE_OUTPUT_PATH"
# Create model-specific output directory
MODEL_DIR="${BASE_OUTPUT_PATH}/${MODEL//\//_}"  # Replace / with _ in model name for safe directory naming
mkdir -p "$MODEL_DIR"

echo "Starting inference for $MODEL with $N_RULES rules and dividing into $N_ROUND"
# Construct input and output paths
INPUT_FILE="${PATH_DATA}/exact_matching_${N_RULES}.json"
OUTPUT_FILE="${MODEL_DIR}/exact_matching_${N_RULES}.json"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
python ../04_FocusNet/hard_seq_run_with_validator.py \
        --input_path="$INPUT_FILE" \
        --template="$TEMPLATE" \
        --output_path="$OUTPUT_FILE" \
        --api_endpoint="$API_ENDPOINT" \
        --llm_model="$MODEL" \
        --h_out="$H_OUT" --h_in="$H_IN" \
        --temperature_focusnet="$TEMPERATURE_RULE_CHECKER" \
        --normalize_embeddings="$NORMALIZE_EMBEDDINGS" \
        --embeddings_aggr="$EMBEDDINGS_AGGR" \
        --pretrained_model "$SENTENCE_ENCODER" \
        --path_prompt_template "$PATH_PROMPT_TEMPLATE" \
        --huggingface_cache="$PATH_HUGGINGFACE_CACHE" \
        --best_model="$VALIDATOR_PATH" \
        --gpu_id="$GPU_ID" --classifier="$CLASSIFIER" \
        --temperature_llm="$TEMPERATURE_LLM"
echo "Completed inference for $MODEL with $N_RULES rules"

