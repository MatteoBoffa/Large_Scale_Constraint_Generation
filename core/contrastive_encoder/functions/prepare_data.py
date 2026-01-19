import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from core.common.utils import read_json
from core.data_extraction.utils import deterministic_hash
from core.llm_encoder.formatter import concatenate_to_template


def tokenize_data(tokenizer, template_file, list_inputs, input_type, sep=" "):
    prompt_template = read_json(path_json=template_file)
    formatted_inputs = [
        concatenate_to_template(prompt_template[input_type], el, sep)
        for el in list_inputs
    ]
    # Tokenize input
    tokenized_input = tokenizer(
        formatted_inputs,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    # Concatenate outputs
    concat_tensors = torch.cat(
        [
            tokenized_input["input_ids"].unsqueeze(1),
            tokenized_input["attention_mask"].unsqueeze(1),
        ],
        dim=1,
    )
    return concat_tensors


def extract_rule_id(rules, sep=" "):
    return deterministic_hash(sep.join(sorted(rules)))


def load_cached_results(cache_dir, logger):
    """
    Attempt to load previously cached positive samples from a pickle file.
    Args:
        cache_dir (str): Directory containing the cache file
        logger: Logger instance for tracking operations
    Returns:
        tuple: Cached results if successful, None if loading fails
    """
    if not cache_dir:
        return None
    cache_file = os.path.join(cache_dir, "positive_samples.pkl")
    if not os.path.exists(cache_file):
        return None
    logger.debug(f"\tLoading cached results from {cache_file}")
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}. Computing from scratch.")
        return None


def get_sample_macro_rules_mapping(df, pos_col, macro_rule_col):
    """
    Create mappings between positive samples and their associated macro rules.

    Args:
        df (DataFrame): Input dataframe containing samples and rules
        pos_col (str): Column name for positive samples
        macro_rule_col (str): Column name for macro rules

    Returns:
        tuple: (pos_sample_to_macro_rules, unique_positive_samples, samples_array)
    """
    # Get unique positive samples and their macro rules
    pos_sample_to_macro_rules = df.groupby(pos_col)[macro_rule_col].apply(set)
    unique_positive_samples = pos_sample_to_macro_rules.index.values
    # Convert sample strings to sets of component rules
    samples_array = np.array(
        [set(sample.split("-")) for sample in unique_positive_samples]
    )
    return pos_sample_to_macro_rules, unique_positive_samples, samples_array


def initialize_result_dictionaries(unique_positive_samples):
    """
    Initialize empty result dictionaries for storing relationships.

    Args:
        unique_positive_samples (array): Array of unique positive sample IDs

    Returns:
        tuple: (real_pos_id_2_macro_rules, pos_id_2_other_positives, set_hashes_x_pos_sample)
    """
    real_pos_id_2_macro_rules = {pos_id: set() for pos_id in unique_positive_samples}
    pos_id_2_other_positives = {pos_id: set() for pos_id in unique_positive_samples}
    set_hashes_x_pos_sample = {}

    return real_pos_id_2_macro_rules, pos_id_2_other_positives, set_hashes_x_pos_sample


def find_subset_relationships(pos_sample_set, samples_array):
    """
    Find indices where pos_sample_set is a subset of other samples.

    Args:
        pos_sample_set (set): Set of rules for current positive sample
        samples_array (array): Array of all sample rule sets

    Returns:
        list: Indices where pos_sample_set is a subset
    """
    return [
        j
        for j, candidate_set in enumerate(samples_array)
        if pos_sample_set.issubset(candidate_set)
    ]


def update_relationships(
    matches,
    unique_positive_samples,
    pos_sample_id,
    pos_sample_to_macro_rules,
    result_dicts,
):
    """
    Update relationship dictionaries based on found matches.

    Args:
        matches (list): Indices of matching samples
        unique_positive_samples (array): Array of unique positive sample IDs
        pos_sample_id (str): Current positive sample ID
        pos_sample_to_macro_rules (Series): Mapping of samples to macro rules
        result_dicts (tuple): Tuple of dictionaries to update
    """
    real_pos_id_2_macro_rules, pos_id_2_other_positives, _ = result_dicts

    for match_idx in matches:
        candidate_id = unique_positive_samples[match_idx]
        real_pos_id_2_macro_rules[pos_sample_id].update(
            pos_sample_to_macro_rules[candidate_id]
        )
        pos_id_2_other_positives[pos_sample_id].add(candidate_id)


def cache_results(results, cache_dir, logger):
    """
    Cache the computed results to a pickle file.

    Args:
        results (tuple): Results to cache
        cache_dir (str): Directory to store cache file
        logger: Logger instance for tracking operations
    """
    if not cache_dir:
        return
    cache_file = os.path.join(cache_dir, "positive_samples.pkl")
    logger.debug(f"\tCaching results to {cache_file}")
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
    except Exception as e:
        logger.warning(f"\tFailed to write cache: {e}")


def extract_positive_samples_2_macro_rules(
    df, pos_col, macro_rule_col, cache_dir, logger
):
    """
    Main function to extract relationships between positive samples and macro rules.

    Args:
        df (DataFrame): Input dataframe containing samples and rules
        pos_col (str): Column name for positive samples
        macro_rule_col (str): Column name for macro rules
        cache_dir (str): Directory for caching results
        logger: Logger instance for tracking operations

    Returns:
        tuple: (real_pos_id_2_macro_rules, pos_id_2_other_positives, set_hashes_x_pos_sample)
    """
    # Try loading cached results first
    cached_results = load_cached_results(cache_dir, logger)
    if cached_results is not None:
        return cached_results
    logger.info("\tComputing positive samples relationships...")
    # Get initial mappings
    pos_sample_to_macro_rules, unique_positive_samples, samples_array = (
        get_sample_macro_rules_mapping(df, pos_col, macro_rule_col)
    )
    # Initialize result dictionaries
    result_dicts = initialize_result_dictionaries(unique_positive_samples)
    real_pos_id_2_macro_rules, pos_id_2_other_positives, set_hashes_x_pos_sample = (
        result_dicts
    )
    # Process each positive sample
    for i, pos_sample_id in tqdm(
        enumerate(unique_positive_samples),
        total=len(unique_positive_samples),
        desc="Obtaining positive pairs...",
    ):
        pos_sample_set = samples_array[i]
        set_hashes_x_pos_sample[pos_sample_id] = pos_sample_set
        # Find and process matches
        matches = find_subset_relationships(pos_sample_set, samples_array)
        update_relationships(
            matches,
            unique_positive_samples,
            pos_sample_id,
            pos_sample_to_macro_rules,
            result_dicts,
        )

    # Cache results
    results = (real_pos_id_2_macro_rules, pos_id_2_other_positives)
    cache_results(results, cache_dir, logger)

    return real_pos_id_2_macro_rules, pos_id_2_other_positives
