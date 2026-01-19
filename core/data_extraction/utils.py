import hashlib
import json
import os
import numpy as np
import pandas as pd
from typing import Any


def export_k_fold_dataset(df, output_dir, fold_idx, partition):
    """
    Export a DataFrame to parquet format and save its characteristics in a specific fold directory structure.

    Args:
        df (pandas.DataFrame): The DataFrame to export
        output_dir (str): Base directory where the fold directories will be created
        fold_idx (int): Index of the current fold
        partition (str): Name of the partition (e.g., 'train', 'test', 'val')

    Returns:
        None: Files are saved to disk:
            - A parquet file at {output_dir}/fold_{fold_idx}/{partition}.parquet
            - A JSON statistics file at {output_dir}/fold_{fold_idx}/{partition}_description.json

    Example:
        >>> df_train = pd.DataFrame(...)
        >>> export_k_fold_dataset(
        ...     df=df_train,
        ...     output_dir='data/processed',
        ...     fold_idx=0,
        ...     partition='train'
        ... )
        # Creates:
        # - data/processed/fold_0/train.parquet
        # - data/processed/fold_0/train_description.json
    """
    if fold_idx is not None:
        folder = os.path.join(output_dir, f"fold_{fold_idx}")
    else:
        folder = output_dir
    os.makedirs(folder, exist_ok=True)
    df.to_parquet(
        os.path.join(folder, f"{partition}.parquet"),
        index=False,
    )
    save_characterization(
        df=df,
        output_file=os.path.join(folder, f"{partition}_description.json"),
    )


def save_characterization(df, output_file):
    """
    Save dataset characterization statistics to a JSON file with nicely formatted numbers
    and statistical descriptions.

    Args:
        df (pandas.DataFrame): DataFrame containing at least the columns
            'positive_sample_id', 'hash_rule_sample'
        output_file (str): Path where the JSON output file will be saved

    Returns:
        None: The statistics are saved to the specified output file

    Example:
        >>> df = pd.DataFrame({
        ...     'positive_sample_id': ['id1', 'id1', 'id2'],
        ...     'hash_rule_sample': ['hash1', 'hash2', 'hash3']
        ... })
        >>> save_characterization(df, 'stats.json')
    """

    # Format describe() output to be more readable
    def format_describe(series_describe):
        stats = series_describe.round(2)
        return {
            "count": f"{stats['count']:,.0f}",
            "mean": f"{stats['mean']:,.2f}",
            "std": f"{stats['std']:,.2f}",
            "min": f"{stats['min']:,.0f}",
            "25%": f"{stats['25%']:,.2f}",
            "50%": f"{stats['50%']:,.2f}",
            "75%": f"{stats['75%']:,.2f}",
            "max": f"{stats['max']:,.0f}",
        }

    # Calculate statistics
    instances_per_rule = (
        df.groupby("positive_sample_id")["hash_rule_sample"].nunique().describe()
    )
    rules_per_instance = df.groupby("hash_rule_sample").size().describe()

    output_dict = {
        "Total rows": f"{df.shape[0]:,}",
        "Unique rules": f"{df.positive_sample_id.nunique():,}",
        "Stats on |rules| x instance": format_describe(rules_per_instance),
        "Stats on |instance| x rules": format_describe(instances_per_rule),
    }

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4)


def weighted_train_val_split(
    df, stratify_col, val_size, weight_col=None, random_state=None
):
    """
    Splits a DataFrame into train and validation sets while ensuring samples with the same
    stratify_col value stay together. Handles cases with duplicate weights by using a more
    robust quartile calculation method.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to split
    stratify_col : str
        Column name to use for grouping samples that should stay together
    val_size : float
        Proportion of data to include in validation split (between 0 and 1)
    weight_col : str, optional
        Column name to use for calculating group weights
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (train_df, val_df) containing the split DataFrames
    """
    # Get unique groups
    unique_groups = df[stratify_col].unique()
    if weight_col is not None:
        # Weight-based strategy
        group_weights = (
            df.groupby(stratify_col)[weight_col].count().reset_index(name="weight")
        )
        try:
            # First attempt: Try qcut with duplicates='drop'
            group_weights["weight_quartile"] = pd.qcut(
                group_weights["weight"],
                q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            )
        except ValueError:
            # Fallback: Manual quartile assignment using quantiles
            quartiles = group_weights["weight"].quantile([0.25, 0.5, 0.75])

            def assign_quartile(x):
                if x <= quartiles[0.25]:
                    return "Q1"
                elif x <= quartiles[0.5]:
                    return "Q2"
                elif x <= quartiles[0.75]:
                    return "Q3"
                else:
                    return "Q4"

            group_weights["weight_quartile"] = group_weights["weight"].apply(
                assign_quartile
            )
        # Initialize random state
        np.random.seed(random_state)
        val_groups = []
        current_val_weight = 0
        # For each quartile
        for quartile in ["Q1", "Q2", "Q3", "Q4"]:
            quartile_groups = group_weights[
                group_weights["weight_quartile"] == quartile
            ].copy()
            if len(quartile_groups) == 0:
                continue
            # Shuffle groups within quartile
            quartile_groups = quartile_groups.sample(frac=1, random_state=random_state)
            # Calculate target weight for this quartile
            quartile_total_weight = quartile_groups["weight"].sum()
            quartile_target_val_weight = quartile_total_weight * val_size
            # Add groups to validation set until target is reached
            quartile_val_weight = 0
            for _, row in quartile_groups.iterrows():
                if quartile_val_weight < quartile_target_val_weight:
                    val_groups.append(row[stratify_col])
                    quartile_val_weight += row["weight"]
                    current_val_weight += row["weight"]
                else:
                    break
        # Remaining groups go to train set
        train_groups = [g for g in unique_groups if g not in val_groups]
    else:
        # Simple random split strategy
        np.random.seed(random_state)
        n_val = int(len(unique_groups) * val_size)
        # Shuffle and split groups
        np.random.shuffle(unique_groups)
        val_groups = unique_groups[:n_val]
        train_groups = unique_groups[n_val:]
    # Split the DataFrame based on the groups
    train_df = df[df[stratify_col].isin(train_groups)].copy()
    val_df = df[df[stratify_col].isin(val_groups)].copy()

    return train_df, val_df


def kfold_split_dataframe(
    df, stratify_col, n_splits, weight_col=None, random_state=None
):
    """
    Splits a DataFrame into K folds while ensuring samples with the same stratify_col value
    stay in the same fold. If weight_col is specified, distributes groups based on weights.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to split
    stratify_col : str
        Column name to use for grouping samples that should stay together
    n_splits : int
        Number of folds for K-fold split
    weight_col : str, optional
        Column name to use for calculating group weights. If None, just splits groups
        randomly while keeping them together
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    list of tuples
        Each tuple contains (train_indices, test_indices) for one fold
    """
    # Reset index to ensure we're working with consecutive integers
    df = df.reset_index(drop=True)
    # Get unique groups
    unique_groups = df[stratify_col].unique()
    if weight_col is not None:
        # Weight-based strategy
        group_weights = (
            df.groupby(stratify_col)[weight_col].count().reset_index(name="weight")
        )
        group_weights = group_weights.sort_values("weight", ascending=False)
        # Initialize random state
        np.random.seed(random_state)
        # Initialize lists to store groups for each fold
        fold_groups = [[] for _ in range(n_splits)]
        fold_sizes = [0] * n_splits
        # Distribute groups across folds based on weights
        for _, row in group_weights.iterrows():
            group = row[stratify_col]
            weight = row["weight"]
            # Find the fold with the smallest current weight
            min_fold_idx = np.argmin(fold_sizes)
            fold_groups[min_fold_idx].append(group)
            fold_sizes[min_fold_idx] += weight
    else:
        # Simple random split strategy
        np.random.seed(random_state)
        # Shuffle the groups
        np.random.shuffle(unique_groups)
        # Split groups into roughly equal-sized folds
        fold_groups = np.array_split(unique_groups, n_splits)
        fold_groups = [group.tolist() for group in fold_groups]

    # Initialize list to store final fold indices
    folds = []
    # Convert fold groups to indices
    for i in range(n_splits):
        # Current fold becomes test set
        test_groups = fold_groups[i]
        # All other folds become training set
        train_groups = [
            group for j, groups in enumerate(fold_groups) for group in groups if j != i
        ]
        # Get indices for samples belonging to these groups
        train_indices = df[df[stratify_col].isin(train_groups)].index
        test_indices = df[df[stratify_col].isin(test_groups)].index

        folds.append((train_indices, test_indices))
    return folds


def combine_datasets(list_datasets):
    """
    Convert a list of HuggingFace datasets into a single pandas DataFrame by concatenating all splits.

    Args:
        list_datasets (datasets.Dataset): The HuggingFace Datasets to be converted

    Returns:
        pandas.DataFrame: A single DataFrame containing all records from all splits

    Example:
        >>> from datasets import load_dataset
        >>> dataset_dict = load_dataset('emotion')
        >>> combined_df = combine_datasets(dataset_dict)
    """
    combined_df = pd.concat(
        [dataset.to_pandas() for dataset in list_datasets],
        ignore_index=True,
    )
    return combined_df


def deterministic_hash(value: Any) -> str:
    """
    Creates a deterministic hash of the input value that remains consistent across different Python sessions.
    Args:
        value (Any): The value to hash. Will be converted to string before hashing.
    Returns:
        str: A hexadecimal string representation of the MD5 hash.
    Examples:
        >>> deterministic_hash("Hello World")
        'b10a8db164e0754105b7a99be72e3fe5'
        >>> deterministic_hash({"a": 1, "b": 2})
        '608de49a4600dbb5b173492759792e4a'
    Notes:
        - Unlike Python's built-in hash() function, this will give the same result across different Python sessions
        - Uses MD5 for speed and consistency, not for cryptographic security
        - If you need cryptographic security, use SHA-256 or another secure hash function
        - The function converts input to string representation before hashing
    """
    # Convert input to string to handle different types
    str_value = str(value)
    # Create MD5 hash of the string value
    hash_object = hashlib.md5(str_value.encode())
    # Return hexadecimal representation of hash
    return hash_object.hexdigest()


def concat_rule_example(sample: pd.Series, column_sample: str, column_rule: str) -> str:
    """
    Concatenates a sentence with its associated rules and creates a deterministic hash.

    Args:
        sample (pandas.Series): A row from a DataFrame containing the sentence and rules
        column_sample (str): Name of the column containing the sentence
        column_rule (str): Name of the column containing the rules

    Returns:
        str: A hexadecimal hash value of the concatenated sentence and rules

    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['This is a sample text'],
        ...     'rules': [['rule1', 'rule2']]
        ... })
        >>> result = concat_rule_example(df.iloc[0], 'text', 'rules')

    Notes:
        - Uses sorted() on the rules to ensure deterministic ordering
        - Removes duplicates while maintaining order using dict.fromkeys()
        - The resulting hash will be consistent across different Python sessions
    """
    # Also handling case with single rules
    list_rules = (
        [sample[column_rule]]
        if isinstance(sample[column_rule], str)
        else sample[column_rule]
    )
    # Remove duplicates while maintaining order and sort for deterministic
    unique_sorted_rules = sorted(dict.fromkeys(list_rules))
    rules = "_".join(unique_sorted_rules)
    sentence = sample[column_sample]
    s = f"{sentence} {rules}"
    return deterministic_hash(s)


def create_hash_rule_sentence(df, column_sample, column_rule):
    """
    Creates a new column 'hash_rule_sample' by applying concat_rule_example to each row.

    Args:
        df (pandas.DataFrame): Input DataFrame containing sentences and rules
        column_sample (str): Name of the column containing sentences
        column_rule (str): Name of the column containing rules

    Returns:
        pandas.DataFrame: DataFrame with new 'hash_rule_sample' column added

    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['Sample 1', 'Sample 2'],
        ...     'rules': [['rule1'], ['rule2']]
        ... })
        >>> result_df = create_hash_rule_sentence(df, 'text', 'rules')
    """
    df_tmp = df.copy()
    df_tmp["hash_rule_sample"] = df_tmp.apply(
        concat_rule_example, args=(column_sample, column_rule), axis=1
    )
    return df_tmp


def create_positive_sample_id(df, rule_column):
    """
    Creates a new column 'positive_sample_id' by hashing values from specified rule column.

    Args:
        df (pandas.DataFrame): Input DataFrame
        rule_column (str): Name of the column to be hashed

    Returns:
        pandas.DataFrame: DataFrame with new 'positive_sample_id' column added

    Example:
        >>> df = pd.DataFrame({'rules': ['rule1', 'rule2']})
        >>> result_df = create_positive_sample_id(df, 'rules')
    """
    # Notice: the value in rule_column can either be a string or a list of single rules.
    # If it's a list, turn it into a set first, so that set's order is constant + remove repetitions.
    df_tmp = df.copy()
    df_tmp["positive_sample_id"] = df_tmp[rule_column].apply(
        lambda el: "-".join(
            [deterministic_hash(concept) for concept in sorted(dict.fromkeys(el))]
        )
    )
    return df_tmp


def explode_column(df, column_2_explode):
    """
    Transforms each element in a list-like column into a separate row.

    Args:
        df (pandas.DataFrame): Input DataFrame
        column_2_explode (str): Name of the column to explode

    Returns:
        pandas.DataFrame: DataFrame with the specified column exploded into separate rows

    Example:
        >>> df = pd.DataFrame({
        ...     'id': [1, 2],
        ...     'tags': [['A', 'B'], ['C']]
        ... })
        >>> exploded_df = explode_column(df, 'tags')
    """
    exploded_df = df.explode(column_2_explode)
    return exploded_df


def select_sample_x_macrorule(df, seed):
    """
    Randomly selects one sample per unique hash_rule_sample group.

    Args:
        df (pandas.DataFrame): Input DataFrame containing 'hash_rule_sample' column
        seed (int): Random seed for reproducible sampling

    Returns:
        pandas.DataFrame: DataFrame containing one randomly selected row per hash_rule_sample group

    Example:
        >>> df = pd.DataFrame({
        ...     'hash_rule_sample': [1, 1, 2, 2],
        ...     'text': ['A', 'B', 'C', 'D']
        ... })
        >>> result_df = select_sample_x_macrorule(df, seed=42)
    """
    sub_exploded_df = df.groupby("hash_rule_sample", group_keys=False).apply(
        lambda x: x.sample(n=1, random_state=seed)
    )
    return sub_exploded_df
