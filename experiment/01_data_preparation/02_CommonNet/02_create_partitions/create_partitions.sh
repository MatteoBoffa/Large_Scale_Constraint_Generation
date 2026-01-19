#!/bin/sh
REPO_ROOT="$(git rev-parse --show-toplevel)"
DATASET_PATH="$REPO_ROOT/offline_datasets/original/CommonNet/commonNet.parquet"
OUTPUT_DIR="$REPO_ROOT/offline_datasets/processed_dataset/CommonNet/kfold_division/"
RULES_COLUMN="concepts"
SAMPLE_SENTENCE="target"
SEED=29
N_SPLITS=5
TEST_SIZE=0.1

# Remember to create the output path if it doesn't exist
mkdir -p $OUTPUT_DIR

# Start the script
python create_partitions.py --dataset_path "$DATASET_PATH"  --output_dir "$OUTPUT_DIR"  --rules_column "$RULES_COLUMN" \
                            --sample_sentence "$SAMPLE_SENTENCE"  --seed "$SEED" --n_splits "$N_SPLITS" \
                            --test_size "$TEST_SIZE"
