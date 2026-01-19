PY="$HOME/bin/python-gpu"   # Only for bigdata cluster > force the working wrapper
#PY="python" # Elsewhere

# Experiment metadata
GPU_ID=0
NUM_PROC=40
SEED=29
REPO_ROOT="$(git rev-parse --show-toplevel)"
EXPERIMENT_ID="no_contrastive_loss_random_forest_training"
PROJECT="rule_constrainer"
GROUP="common_net"
LOGGER_VERBOSITY="info"
USE_CACHE="on"

# Input data paths
FOLD=0 # Hardcoded fold number for ablation study
TRAINING_DATA_PATH="$REPO_ROOT/offline_datasets/processed_dataset/CommonNet/kfold_division/fold_$FOLD/single_rule_train.parquet"
VALIDATION_DATA_PATH="$REPO_ROOT/offline_datasets/processed_dataset/CommonNet/kfold_division/fold_$FOLD/single_rule_validation.parquet"
INPUT_PATHS_2_ENCODE=($TRAINING_DATA_PATH $VALIDATION_DATA_PATH)
FILE_IDS=("train" "validation")
PATH_PROMPT_TEMPLATE="../prompts/prompt_templates.json"
PATH_HUGGINGFACE_CACHE="$REPO_ROOT/hf_cache"

# Output data paths
OUTPUT_PATH="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID"
PATH_CACHE_INTERMIDIATE_RESULTS="$OUTPUT_PATH/cache_intermidiate_datasets/"
#[ -d "$PATH_CACHE_INTERMIDIATE_RESULTS" ] && rm -r "$PATH_CACHE_INTERMIDIATE_RESULTS"
#mkdir -p $PATH_CACHE_INTERMIDIATE_RESULTS
FOLD_OUTPUT_PATH="$OUTPUT_PATH/training_best_model" 
#[ -d "$FOLD_OUTPUT_PATH" ] && rm -r "$FOLD_OUTPUT_PATH"
#mkdir -p $FOLD_OUTPUT_PATH
#################### DATASET COLUMN NAMES
POS_SAMPLE_COLUMN="positive_sample_id"
COLUMN_MACRO_RULE_ID="hash_rule_sample"
RULE_COLUMN="concepts"
INPUT_COLUMN="target"
#################### ENCODER 
SENTENCE_ENCODER="sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE=256
NORMALIZE_EMBEDDINGS="True"
EMBEDDINGS_AGGR="mean_pooling"
#################### CLASSIFICATION HEAD TRAINING
CLASSIFIER="rf"

echo "########################################################################################## PHASE I - Train a classification head"

#Train classification head only without contrastive loss
# CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" ../04b_train_classification_head_only.py --experiment_id="$EXPERIMENT_ID" --seed="$SEED"  \
#                         --logger_verbosity="$LOGGER_VERBOSITY" --seed="$SEED" --cache_dir="$PATH_CACHE_INTERMIDIATE_RESULTS" \
#                         --path_training_data="$TRAINING_DATA_PATH" --path_validation_data="$VALIDATION_DATA_PATH" --path_prompt_template="$PATH_PROMPT_TEMPLATE" \
#                         --pos_sample_column="$POS_SAMPLE_COLUMN" --macro_rule_id_column="$COLUMN_MACRO_RULE_ID" --output_path="$FOLD_OUTPUT_PATH" \
#                         --num_proc "$NUM_PROC" --batch_size="$BATCH_SIZE" --rule_column="$RULE_COLUMN" --input_column="$INPUT_COLUMN" \
#                         --pretrained_model="$SENTENCE_ENCODER" --normalize_embeddings="$NORMALIZE_EMBEDDINGS" \
#                         --embeddings_aggr="$EMBEDDINGS_AGGR" --huggingface_cache="$PATH_HUGGINGFACE_CACHE" --classifier="$CLASSIFIER" 

echo "########################################################################################## PHASE II - Tune threshold to maximize recall"

VALIDATION_SET="$REPO_ROOT/offline_datasets/processed_dataset/CommonNet/kfold_division/fold_0/single_rule_validation.parquet"
CLASSIFIER="rf"
MODEL_PATH="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID/training_best_model/best_model"
BATCH_SIZE=512
PROB_NEG_SAMPLE=0.5
HAS_CONTRASTIVE="False"

PATH_TENSORBOARD_LOGS="$OUTPUT_PATH/tensorboard_logs/"
PATH_LOGGING_DATA="$PATH_TENSORBOARD_LOGS/best_model/"
# Make sure we have clean logs for each run
[ -d "$PATH_LOGGING_DATA" ] && rm -r "$PATH_LOGGING_DATA"
mkdir -p $PATH_LOGGING_DATA

CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" ../05_tune_threshold.py --experiment_id="$EXPERIMENT_ID" --seed="$SEED" \
                        --logger_verbosity="$LOGGER_VERBOSITY" --path_test_data="$VALIDATION_SET" --logging_path="$PATH_LOGGING_DATA" \
                        --path_prompt_template "$PATH_PROMPT_TEMPLATE" --pretrained_model "$SENTENCE_ENCODER" \
                        --rule_column="$RULE_COLUMN" --input_column="$INPUT_COLUMN" --huggingface_cache="$PATH_HUGGINGFACE_CACHE" \
                        --normalize_embeddings="$NORMALIZE_EMBEDDINGS" --embeddings_aggr="$EMBEDDINGS_AGGR" --best_model="$MODEL_PATH" \
                        --pos_sample_column="$POS_SAMPLE_COLUMN" --macro_rule_id_column="$COLUMN_MACRO_RULE_ID" \
                        --prob_neg_sample="$PROB_NEG_SAMPLE" --num_proc="$NUM_PROC"  \
                        --normalize_embeddings="$NORMALIZE_EMBEDDINGS" --classifier="$CLASSIFIER" \
                        --batch_size="$BATCH_SIZE" --has_contrastive="$HAS_CONTRASTIVE"
