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
BASE_OUTPUT_PATH="$REPO_ROOT/experiment_results/WordChecker/one_shot"
# Create base output directory
mkdir -p "$BASE_OUTPUT_PATH"

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

echo "Input file: $INPUT_FILE"
echo "Output directory: $COMPLETE_OUTPUT"
python ../01_simple_prompt/one_shot_inference.py \
    --input_path="$INPUT_FILE" \
    --template="$TEMPLATE" \
    --output_path="$COMPLETE_OUTPUT" \
    --api_endpoint="$API_ENDPOINT" \
    --model_name="$MODEL" \
    --temperature="$TEMPERATURE" \
    --judge_model="$JUDGE_MODEL" \
    --nltk_data_dir="$PATH_NLTK_CACHE"
echo "Completed inference for $MODEL with $N_RULES rules"
