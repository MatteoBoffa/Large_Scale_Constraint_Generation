PY="$HOME/bin/python-gpu"   # Only for bigdata cluster > force the working wrapper
#PY="python" # Elsewhere

# Experiment metadata
GPU_ID=0
NUM_PROC=10
H_OUTS=(128)
LOSSES_RULES=("True")
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Iterate over the different loss rules
for LOSS_RULES in "${LOSSES_RULES[@]}"; do
# Iterate over the different H_OUTS
    for H_OUT in "${H_OUTS[@]}"; do
        EXPERIMENT_ID="random_forest_training_hout_${H_OUT}_loss_rules_${LOSS_RULES}"
        PROJECT="rule_constrainer"
        GROUP="common_gen"
        OUTPUT_PATH="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID"
        LOGGER_VERBOSITY="info"
        USE_CACHE="on" #["on", "off"]
        SEED=1
        FOLDS=(0 1 2 3) # We will only use the first fold
        ########################################### TRAIN, ITERATING OVER FOLDS
        echo "########################################################################################## PHASE I - Train"
        for FOLD in "${FOLDS[@]}"; do
            echo "############################################# Training fold $FOLD #############################################"
            ########################################################################################## PHASE I - ENCODING
            #################### IO
            TRAINING_DATA_PATH="$REPO_ROOT/offline_datasets/processed_dataset/CommonGen/kfold_division/fold_$FOLD/train.parquet"
            VALIDATION_DATA_PATH="$REPO_ROOT/offline_datasets/processed_dataset/CommonGen/kfold_division/fold_$FOLD/validation.parquet"
            INPUT_PATHS_2_ENCODE=($TRAINING_DATA_PATH $VALIDATION_DATA_PATH)
            FILE_IDS=("train" "validation")
            PATH_PROMPT_TEMPLATE="../prompts/prompt_templates.json"
            PATH_HUGGINGFACE_CACHE="$REPO_ROOT/hf_cache"
            FOLD_OUTPUT_PATH="$OUTPUT_PATH/fold_$FOLD" 
            #################### DATASET COLUMN NAMES
            RULE_COLUMN="concepts"
            INPUT_COLUMN="target"
            #################### ENCODER 
            SENTENCE_ENCODER="sentence-transformers/all-mpnet-base-v2"
            BATCH_SIZE=256
            NORMALIZE_EMBEDDINGS="True"
            EMBEDDINGS_AGGR="mean_pooling"
            # Check if the output directory exists + if caching is 
            # if [ -d "$FOLD_OUTPUT_PATH/intermidiate_datasets/embeddings" ] && [ "$USE_CACHE" == "on" ]; then
            #     echo "Embeddings already exist: we will use the cached values"
            # else
            #     # Remove the output folder if it already exists and cache is not allowed
            #     echo "Creating the embedding from scratch!"
            #     if [ -d "$FOLD_OUTPUT_PATH" ]; then
            #         echo "Removing previous experiments at: `$FOLD_OUTPUT_PATH`..."
            #         rm -r "$FOLD_OUTPUT_PATH"
            #     fi
            #     # Create a "clean" output directory
            #     mkdir -p $FOLD_OUTPUT_PATH
            #     CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" ../01_embeddings_extractor.py --logger_verbosity "$LOGGER_VERBOSITY" \
            #                                     --file_paths "${INPUT_PATHS_2_ENCODE[@]}" --file_ids "${FILE_IDS[@]}" \
            #                                     --output_path "$FOLD_OUTPUT_PATH" --prompt_template "$PATH_PROMPT_TEMPLATE" \
            #                                     --sentence_encoder "$SENTENCE_ENCODER" --rule_column="$RULE_COLUMN" \
            #                                     --input_column="$INPUT_COLUMN" --batch_size="$BATCH_SIZE" --huggingface_cache="$PATH_HUGGINGFACE_CACHE" \
            #                                     --normalize_embeddings="$NORMALIZE_EMBEDDINGS" --embeddings_aggr="$EMBEDDINGS_AGGR"
            # fi
            ########################################################################################## PHASE II - RULE TRAINER
            #################### FURTHER EXPERIMENT METADATA
            PATH_LOGGING_DATA="$OUTPUT_PATH/tensorboard_logs/fold_$FOLD/" 
            #################### FURTHER DATASET COLUMN NAMES
            POS_SAMPLE_COLUMN="positive_sample_id"
            COLUMN_MACRO_RULE_ID="hash_rule_sample"
            #################### FURTHER IO 
            PROCESSED_TRAINING_DATA_PATH="$FOLD_OUTPUT_PATH/intermidiate_datasets/preprocessed_data/train.parquet"
            PATH_TRAINING_EMBEDDINGS="$FOLD_OUTPUT_PATH/intermidiate_datasets/embeddings/train/"
            PROCESSED_VALIDATION_DATA_PATH="$FOLD_OUTPUT_PATH/intermidiate_datasets/preprocessed_data/validation.parquet"
            PATH_VALIDATION_EMBEDDINGS="$FOLD_OUTPUT_PATH/intermidiate_datasets/embeddings/validation/"
            # NOTICE: since the generation of the negative samples is part of a random process, the cached values will also be seed dependant
            PATH_CACHE_INTERMIDIATE_RESULTS="$FOLD_OUTPUT_PATH/intermidiate_datasets/preprocessed_data/seed_$SEED/"
            MODELS_OUTPUT_PATH="$FOLD_OUTPUT_PATH/rule_constr_training"
            #################### MODEL'S TRAINING HYPERPARAMETERS
            EPOCHS=30
            NS=(1) # Number of negative samples fixed to 3
            H_IN=768
            LRS=(2.5e-4)
            TEMPERATURES=(0.05)
            BATCH_SIZE=384
            LR_SCHEDULER="linear"

            # for N_NEGATIVE_SAMPLES in "${NS[@]}"; do
            #     for TEMPERATURE in "${TEMPERATURES[@]}"; do
            #         for LR in "${LRS[@]}"; do
            #             HYPERPARAMETERS="ns_${N_NEGATIVE_SAMPLES}_lr_${LR}_hout_${H_OUT}_scheduler_${LR_SCHEDULER}_temp_${TEMPERATURE}"
            #             ID="${EXPERIMENT_ID}_seed_${SEED}_${HYPERPARAMETERS}"
            #             LOGGING_PATH="${PATH_LOGGING_DATA}/seed_${SEED}/${HYPERPARAMETERS}"
            #             CONSTRAINER_OUTPUT_PATH="${MODELS_OUTPUT_PATH}/seed_${SEED}/${HYPERPARAMETERS}"
            #             # Make sure we have clean logs for each run
            #             [ -d "$LOGGING_PATH" ] && rm -r "$LOGGING_PATH"
            #             mkdir -p $LOGGING_PATH
            #             if [ "$USE_CACHE" == "off" ]; then
            #                 # Don't use cache or training checkpoints, if cache is not allowed
            #                 [ -d "$PATH_CACHE_INTERMIDIATE_RESULTS" ] && rm -r "$PATH_CACHE_INTERMIDIATE_RESULTS"
            #                 [ -d "$CONSTRAINER_OUTPUT_PATH" ] && rm -r "$CONSTRAINER_OUTPUT_PATH"
            #             fi
            #             mkdir -p $CONSTRAINER_OUTPUT_PATH
            #             mkdir -p $PATH_CACHE_INTERMIDIATE_RESULTS
            #             # Starts running
            #             CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" ../02_contrastive_encoder_train.py --experiment_id="$ID" --seed="$SEED"  \
            #                                     --logging_path="$LOGGING_PATH" --logger_verbosity="$LOGGER_VERBOSITY" \
            #                                     --path_training_data="$PROCESSED_TRAINING_DATA_PATH" --path_training_embeddings="$PATH_TRAINING_EMBEDDINGS" \
            #                                     --path_validation_data="$PROCESSED_VALIDATION_DATA_PATH" --path_validation_embeddings="$PATH_VALIDATION_EMBEDDINGS" \
            #                                     --pos_sample_column="$POS_SAMPLE_COLUMN" --macro_rule_id_column="$COLUMN_MACRO_RULE_ID" --output_path="$CONSTRAINER_OUTPUT_PATH" \
            #                                     --epochs="$EPOCHS" --negative_samples="$N_NEGATIVE_SAMPLES" --lr="$LR" --h_out="$H_OUT" --h_in="$H_IN" \
            #                                     --num_proc "$NUM_PROC" --cache_dir "$PATH_CACHE_INTERMIDIATE_RESULTS" --batch_size="$BATCH_SIZE" --lr_scheduler="$LR_SCHEDULER" \
            #                                     --temperature "$TEMPERATURE" --loss_rules "$LOSS_RULES"
            #         done
            #     done
            # done
        done
        echo "########################################################################################## PHASE II - Extracting best hyperparameters"

        PATH_TENSORBOARD_LOGS="$OUTPUT_PATH/tensorboard_logs/"
        METRIC_UNDER_ANALYSIS="control/MRI@1"
        python_output=$("$PY" ../03_best_hyperparameters_extractor.py --log_path "$PATH_TENSORBOARD_LOGS" --metric_analysis "$METRIC_UNDER_ANALYSIS")
        # Parse individual values using jq
        read -r best_epoch best_ns best_lr best_h_out best_scheduler best_temperature <<< $(echo $python_output | jq -r '[.epoch, .ns, .lr, .h_out, .scheduler, .temperature] | @tsv')
        echo "Best Epoch: $best_epoch, Best NS: $best_ns, Best LR: $best_lr, Best H_OUT: $best_h_out, Best Scheduler: $best_scheduler, Best Temperature: $best_temperature"
        echo "########################################################################################## PHASE III - Training model with best hyper-parameters"
        SEED=29 #Fixing seed here, as we shall already considered randomicity
        PATH_LOGGING_DATA="$PATH_TENSORBOARD_LOGS/best_model/"
        # Make sure we have clean logs for each run
        [ -d "$PATH_LOGGING_DATA" ] && rm -r "$PATH_LOGGING_DATA"
        mkdir -p $PATH_LOGGING_DATA
        FOLD_OUTPUT_PATH="$OUTPUT_PATH/fold_0" # Use the first seed, as we will use the concatenation of training and validation for the final training
        PROCESSED_TRAINING_DATA_PATH="$FOLD_OUTPUT_PATH/intermidiate_datasets/preprocessed_data/train.parquet"
        PATH_TRAINING_EMBEDDINGS="$FOLD_OUTPUT_PATH/intermidiate_datasets/embeddings/train/"
        PROCESSED_VALIDATION_DATA_PATH="$FOLD_OUTPUT_PATH/intermidiate_datasets/preprocessed_data/validation.parquet"
        PATH_VALIDATION_EMBEDDINGS="$FOLD_OUTPUT_PATH/intermidiate_datasets/embeddings/validation/"
        PATH_CACHE_INTERMIDIATE_RESULTS="$OUTPUT_PATH/cache_intermidiate_datasets/"
        BEST_OUTPUT_PATH="$OUTPUT_PATH/training_best_model"
        POS_SAMPLE_COLUMN="positive_sample_id"
        COLUMN_MACRO_RULE_ID="hash_rule_sample"
        H_IN=768
        BATCH_SIZE=128
        # CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" ../04_best_run_contrastive_encoder_train.py --experiment_id="$EXPERIMENT_ID" --seed="$SEED"  \
        #                         --logging_path="$PATH_LOGGING_DATA" --logger_verbosity="$LOGGER_VERBOSITY" \
        #                         --path_training_data="$PROCESSED_TRAINING_DATA_PATH" --path_training_embeddings="$PATH_TRAINING_EMBEDDINGS" \
        #                         --path_validation_data="$PROCESSED_VALIDATION_DATA_PATH" --path_validation_embeddings="$PATH_VALIDATION_EMBEDDINGS" \
        #                         --pos_sample_column="$POS_SAMPLE_COLUMN" --macro_rule_id_column="$COLUMN_MACRO_RULE_ID" --output_path="$BEST_OUTPUT_PATH" \
        #                         --epochs="$best_epoch" --negative_samples="$best_ns" --lr="$best_lr" --h_out="$best_h_out" --h_in="$H_IN" \
        #                         --num_proc "$NUM_PROC" --cache_dir "$PATH_CACHE_INTERMIDIATE_RESULTS" --batch_size="$BATCH_SIZE" --lr_scheduler="$best_scheduler" \
        #                         --temperature "$best_temperature" --loss_rules "$LOSS_RULES"

        echo "########################################################################################## PHASE IV - Tune threshold to maximize recall"

        VALIDATION_SET="$REPO_ROOT/offline_datasets/processed_dataset/CommonNet/kfold_division/fold_0/validation.parquet"
        CLASSIFIER="rf"
        MODEL_PATH="$REPO_ROOT/experiment_results/$PROJECT/$GROUP/$EXPERIMENT_ID/training_best_model/best_model"
        BATCH_SIZE=512
        PROB_NEG_SAMPLE=0.5

        CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" ../05_tune_threshold.py --experiment_id="$EXPERIMENT_ID" --seed="$SEED" \
                                --logger_verbosity="$LOGGER_VERBOSITY" --path_test_data="$VALIDATION_SET" --logging_path="$PATH_LOGGING_DATA" \
                                --path_prompt_template "$PATH_PROMPT_TEMPLATE" --pretrained_model "$SENTENCE_ENCODER" \
                                --rule_column="$RULE_COLUMN" --input_column="$INPUT_COLUMN" --huggingface_cache="$PATH_HUGGINGFACE_CACHE" \
                                --normalize_embeddings="$NORMALIZE_EMBEDDINGS" --embeddings_aggr="$EMBEDDINGS_AGGR" --best_model="$MODEL_PATH" \
                                --pos_sample_column="$POS_SAMPLE_COLUMN" --macro_rule_id_column="$COLUMN_MACRO_RULE_ID" \
                                --prob_neg_sample="$PROB_NEG_SAMPLE" --h_out="$H_OUT" --h_in="$H_IN" --num_proc="$NUM_PROC"  \
                                --normalize_embeddings="$NORMALIZE_EMBEDDINGS" --temperature "$best_temperature" --classifier="$CLASSIFIER" \
                                --batch_size="$BATCH_SIZE" 

    done
done