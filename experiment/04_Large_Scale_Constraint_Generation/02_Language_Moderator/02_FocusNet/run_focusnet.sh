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
#GPU_ID
GPU_ID=0

PATH_DATA="$REPO_ROOT/offline_datasets/processed_dataset/CommonGen/test_many_rules"
BASE_OUTPUT_PATH="$REPO_ROOT/experiment_results/WordChecker/FocusNet"
# Create base output directory
mkdir -p "$BASE_OUTPUT_PATH"

# Details about focusnet
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
PATH_PROMPT_TEMPLATE="../02_FocusNet/prompts/prompt_templates.json"
PATH_HUGGINGFACE_CACHE="$REPO_ROOT/hf_cache"
EXPERIMENT_ID="knn_training_hout_${H_OUT}_loss_rules_${LOSS_RULES}"
VALIDATOR_PATH="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID/training_best_model/best_model"

# Configuration
TEMPERATURE=0.4
TEMPLATE="Your task is to validate or rewrite a sentence according to a list of forbidden words.

You are given:
1) A sentence as input.
2) A list of forbidden words.

The sentence must NOT contain:
- Any forbidden word
- Any lemma, stem, or morphological variant of a forbidden word

Instructions:
- First, check whether the input sentence contains any forbidden word, stem, or lemma.
- If the sentence is valid, return the exact same sentence.
- If the sentence is NOT valid, generate a new sentence that is semantically equivalent
  (i.e., it preserves the same meaning and intent) but does NOT contain any forbidden
  words, stems, or lemmas.
- You may use synonyms or rephrasing if necessary.

Output format (mandatory):
<answer>Your sentence</answer>"
JUDGE_MODEL="gpt-4.1"
PATH_NLTK_CACHE="$REPO_ROOT/nltk_cache"

# Create model-specific output directory
MODEL_DIR="${BASE_OUTPUT_PATH}/${MODEL//\//_}"  # Replace / with _ in model name for safe directory naming
mkdir -p "$MODEL_DIR"
echo "Starting inference for $MODEL with $N_RULES rules"
# Construct input and output paths
INPUT_FILE="${PATH_DATA}/exact_matching_${N_RULES}.json"
COMPLETE_OUTPUT="${MODEL_DIR}/correcting_${N_RULES}"
mkdir -p "$COMPLETE_OUTPUT"

# Create output directory if it doesn't exist
echo "Input file: $INPUT_FILE"
echo "Output directory: $COMPLETE_OUTPUT"
python ../02_FocusNet/focusnet_inference.py \
    --input_path="$INPUT_FILE" \
    --template="$TEMPLATE" \
    --output_path="$COMPLETE_OUTPUT" \
    --api_endpoint="$API_ENDPOINT" \
    --model_name="$MODEL" \
    --temperature="$TEMPERATURE" \
    --judge_model="$JUDGE_MODEL" \
    --nltk_data_dir="$PATH_NLTK_CACHE" \
    --h_out="$H_OUT" --h_in="$H_IN" \
    --temperature_focusnet="$TEMPERATURE_RULE_CHECKER" \
    --normalize_embeddings="$NORMALIZE_EMBEDDINGS" \
    --embeddings_aggr="$EMBEDDINGS_AGGR" \
    --pretrained_model "$SENTENCE_ENCODER" \
    --path_prompt_template "$PATH_PROMPT_TEMPLATE" \
    --huggingface_cache="$PATH_HUGGINGFACE_CACHE" \
    --best_model="$VALIDATOR_PATH" \
    --gpu_id="$GPU_ID" --classifier="$CLASSIFIER" 
echo "Completed inference for $MODEL with $N_RULES rules"
