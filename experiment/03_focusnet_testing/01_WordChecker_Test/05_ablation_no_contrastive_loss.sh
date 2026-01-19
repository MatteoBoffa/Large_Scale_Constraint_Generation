#!/bin/bash

PY="$HOME/bin/python-gpu"   # Only for bigdata cluster > force the working wrapper
#PY="python" # Elsewhere
REPO_ROOT="$(git rev-parse --show-toplevel)"

# HARDWARE SETTINGS
GPU_ID=0
NUM_PROC=30
SEED=42

# HERE, PUT THE MODEL WE WANT TO TEST
EXPERIMENT_ID="no_contrastive_loss_random_forest_training"
GROUP="common_net"
PROJECT="rule_constrainer"

echo "########################################################################################## Test trained model on CommonNet single-rule testing dataset"
### GENERAL
LOGGER_VERBOSITY="info"
PROB_NEG_SAMPLE=0.5 # With this number, we set the negative samples (in this case, we double the number of rows)
BATCH_SIZE=512

### OUTPUT DATA
PATH_LOGGING_DATA="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID/test_results/single_rules_common_gen_test"
# Make sure we have clean logs for each run
[ -d "$PATH_LOGGING_DATA" ] && rm -r "$PATH_LOGGING_DATA"
mkdir -p $PATH_LOGGING_DATA

### INPUT DATA
TEST_DATA_PATH="$REPO_ROOT/offline_datasets/processed_dataset/CommonNet/kfold_division/single_rules_test_set.parquet"
RULE_COLUMN="concepts"
INPUT_COLUMN="target"
COLUMN_MACRO_RULE_ID="hash_rule_sample"
POS_SAMPLE_COLUMN="positive_sample_id"
### MODEL
MODEL_PATH="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID/training_best_model/best_model"
NORMALIZE_EMBEDDINGS="True"
SENTENCE_ENCODER="sentence-transformers/all-mpnet-base-v2" #"sentence-transformers/all-mpnet-base-v2" # 
EMBEDDINGS_AGGR="mean_pooling"
PATH_PROMPT_TEMPLATE="prompts/prompt_templates.json"
PATH_HUGGINGFACE_CACHE="$REPO_ROOT/hf_cache"
### PARAMETERS OF BEST MODEL
CLASSIFIER="rf"  # Options: knn, random_forest
HAS_CONTRASTIVE="False"

CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" ./test.py --experiment_id="$EXPERIMENT_ID" --seed="$SEED" \
                        --logger_verbosity="$LOGGER_VERBOSITY" --path_test_data="$TEST_DATA_PATH" --logging_path="$PATH_LOGGING_DATA" \
                        --path_prompt_template "$PATH_PROMPT_TEMPLATE" --pretrained_model "$SENTENCE_ENCODER" \
                        --rule_column="$RULE_COLUMN" --input_column="$INPUT_COLUMN" --huggingface_cache="$PATH_HUGGINGFACE_CACHE" \
                        --normalize_embeddings="$NORMALIZE_EMBEDDINGS" --embeddings_aggr="$EMBEDDINGS_AGGR" --best_model="$MODEL_PATH" \
                        --pos_sample_column="$POS_SAMPLE_COLUMN" --macro_rule_id_column="$COLUMN_MACRO_RULE_ID" \
                        --prob_neg_sample="$PROB_NEG_SAMPLE" --num_proc="$NUM_PROC"  \
                        --normalize_embeddings="$NORMALIZE_EMBEDDINGS" --batch_size="$BATCH_SIZE" \
                        --classifier="$CLASSIFIER" --has_contrastive="$HAS_CONTRASTIVE"