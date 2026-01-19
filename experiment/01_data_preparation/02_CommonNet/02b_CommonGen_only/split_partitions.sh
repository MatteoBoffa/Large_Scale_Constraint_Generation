#!/bin/sh
REPO_ROOT="$(git rev-parse --show-toplevel)"

DATASET_PATH="GEM/common_gen"
CACHE_DIR="$REPO_ROOT/offline_datasets/original/CommonGen/"
OUTPUT_DIR="$REPO_ROOT/offline_datasets/processed_dataset/CommonGen/kfold_division"
TRAINING_PARTITION="train"
VALIDATION_PARTITION="validation"
TEST_PARTITION="challenge_validation_sample"
RULES_COLUMN="concepts"
SAMPLE_SENTENCE="target"
SEED=29
N_SPLITS=5
TEST_SIZE=0.1

# Remember to create the output path if it doesn't exist
mkdir -p $OUTPUT_DIR

# Start the script
python split_partitions.py --dataset_path "$DATASET_PATH" --cache_dir "$CACHE_DIR" \
                                  --output_dir "$OUTPUT_DIR" --training_partition "$TRAINING_PARTITION" \
                                  --validation_partition "$VALIDATION_PARTITION" --testing_partition "$TEST_PARTITION" \
                                  --rules_column "$RULES_COLUMN" --sample_sentence "$SAMPLE_SENTENCE"  --seed "$SEED" \
                                  --n_splits "$N_SPLITS" --test_size "$TEST_SIZE"