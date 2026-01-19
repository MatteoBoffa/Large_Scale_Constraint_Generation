from core.data_extraction.parser import get_extraction_parameters
from core.common.utils import load_dataframe
from core.data_extraction.utils import (
    create_hash_rule_sentence,
    create_positive_sample_id,
    explode_column,
    export_k_fold_dataset,
    kfold_split_dataframe,
    weighted_train_val_split,
)


def pre_process_data(
    dataset_path,
    output_dir,
    rules_column,
    sample_sentence,
    training_partition,
    validation_partition,
    testing_partition,
    seed,
    n_splits,
    test_size,
    **kwargs,
):
    # 1. Load the dataset
    print("Loading the original data...")
    df = load_dataframe(dataset_path)
    # 2. Create a sample id for each single_rule (id identifying the tuple ["target", "concepts"])
    print("Creating hash to identify unique tuples (sentence, [rule1, rule2, etc.])...")
    hashed_dataset = create_hash_rule_sentence(df, sample_sentence, rules_column)
    # 3. To better identify positive and negative samples, create a `positive_sample_ID`.
    #   Such ID will be the same for all target sentences sharing the same single rule.
    print(
        "Creating an hash to identify positive examples (rows sharing the same single rule)..."
    )
    positive_samples_df = create_positive_sample_id(hashed_dataset, rules_column)
    # 4. Explode the macro-rules column: each macro-rule must become a single-rule dataset
    print("Exploding `rules_column` > each single rule becomes a row...")
    pre_processed_df = explode_column(positive_samples_df, rules_column)
    # 5. Extract training and test sets
    #   N.b. We want that samples of the same rule don't go in different partitions
    #   At the same time, let's spreading common and rare rules across all folds
    print(
        f"Get train and test sets with % ratio of {(1-test_size)*100:.0f}vs{(test_size)*100:.0f}"
    )
    print("\tNotice: train and test will contain disjoint rules")
    train_df, test_df = weighted_train_val_split(
        pre_processed_df,
        val_size=test_size,
        stratify_col="positive_sample_id",
        random_state=seed,
        weight_col="hash_rule_sample",
    )
    train_df = train_df.reset_index(drop=True)
    # 7. Now, obtain train and validation following a K-Fold strategy
    #   N.b. We want that samples of the same rule don't go in different partitions
    #   At the same time, let's spreading common and rare rules across all folds
    print(f"Splitting the training dataset into {n_splits} folds...")
    folds = kfold_split_dataframe(
        df=train_df,
        stratify_col="positive_sample_id",
        weight_col="hash_rule_sample",
        n_splits=n_splits,
        random_state=seed,
    )
    # 8. Saving partitions
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        print(f"Saving fold {fold_idx}...", end="\r")
        final_train_df = train_df.iloc[train_indices].copy()
        final_valid_df = train_df.iloc[val_indices].copy()
        # 8a. Save the output dataframes
        export_k_fold_dataset(
            df=final_train_df,
            output_dir=output_dir,
            fold_idx=fold_idx,
            partition=training_partition,
        )
        export_k_fold_dataset(
            df=final_valid_df,
            output_dir=output_dir,
            fold_idx=fold_idx,
            partition=validation_partition,
        )
    export_k_fold_dataset(
        df=test_df,
        output_dir=output_dir,
        fold_idx=None,
        partition=testing_partition,
    )


if __name__ == "__main__":
    args = get_extraction_parameters()
    pre_process_data(**vars(args))
