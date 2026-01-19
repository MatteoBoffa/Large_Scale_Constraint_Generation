#!/bin/bash

echo "########################################################################################## Assess whether trained model can assign concepts to the corresponding specific definitions"

PY="$HOME/bin/python-gpu"   # Only for bigdata cluster > force the working wrapper
#PY="python" # Elsewhere
REPO_ROOT="$(git rev-parse --show-toplevel)"
LOGGER_VERBOSITY="info"

# HARDWARE SETTINGS
GPU_ID=0
NUM_PROC=30
SEED=42

# HERE, PUT THE MODEL WE WANT TO TEST
EXPERIMENT_ID="no_contrastive_loss_random_forest_training"
GROUP="common_net"
PROJECT="rule_constrainer"

### OUTPUT DATA
OUTPUT_PATH="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID/test_results/wordnet_definitions/specific_definitions"
# Make sure we have clean logs for each run
[ -d "$OUTPUT_PATH" ] && rm -r "$OUTPUT_PATH"
mkdir -p $OUTPUT_PATH
### INPUT DATA
TEST_DATA_PATH="$REPO_ROOT/offline_datasets/processed_dataset/WordNet/rules_vs_definitions/specific_concepts_vs_definitions.json"
### MODEL
MODEL_PATH="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID/training_best_model/best_model"
NORMALIZE_EMBEDDINGS="True"
SENTENCE_ENCODER="sentence-transformers/all-mpnet-base-v2" #"sentence-transformers/all-mpnet-base-v2" # 
EMBEDDINGS_AGGR="mean_pooling"
PATH_PROMPT_TEMPLATE="../prompts/prompt_templates.json"
PATH_HUGGINGFACE_CACHE="$REPO_ROOT/hf_cache"
### PARAMETERS OF BEST MODEL
H_OUT=128
H_IN=768    
TEMPERATURE=0.05

CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" ../check_definition.py --experiment_id="$EXPERIMENT_ID" --logging_path="$OUTPUT_PATH" --num_proc="$NUM_PROC" \
                        --logger_verbosity="$LOGGER_VERBOSITY" --path_test_data="$TEST_DATA_PATH"  --best_model="$MODEL_PATH" \
                        --path_prompt_template "$PATH_PROMPT_TEMPLATE" --pretrained_model "$SENTENCE_ENCODER" \
                        --normalize_embeddings="$NORMALIZE_EMBEDDINGS" --embeddings_aggr="$EMBEDDINGS_AGGR" \
                        --huggingface_cache="$PATH_HUGGINGFACE_CACHE" --h_out="$H_OUT" --h_in="$H_IN" \
                        --normalize_embeddings="$NORMALIZE_EMBEDDINGS" --temperature "$TEMPERATURE" 