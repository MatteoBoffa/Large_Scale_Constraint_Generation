import argparse
import json
import os
import random
from core.common.data_handler import pad_ragged_tensor_lists
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from core.common.utils import format_with_hashes, get_logger, read_json, set_seeds
from core.contrastive_encoder.classes.rule_encoder import (
    ContrastiveConstrainerModel,
    NaiveConstrainerModel,
)
from core.contrastive_encoder.functions.metrics import compute_mri
from core.contrastive_encoder.functions.prepare_data import tokenize_data


def parse_arguments():
    """
    Parse command line arguments for the test script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Parse arguments for test script")

    # Required arguments
    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="Unique identifier for the experiment",
    )
    parser.add_argument(
        "--seed", type=int, default=29, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--logger_verbosity", type=str, required=True, help="Logging verbosity level"
    )
    parser.add_argument(
        "--path_test_data", type=str, required=True, help="Path to test data"
    )
    parser.add_argument(
        "--logging_path", type=str, required=True, help="Path for storing logs"
    )
    parser.add_argument(
        "--path_prompt_template",
        type=str,
        required=True,
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Name or path of the pretrained sentence encoder model",
    )
    parser.add_argument(
        "--best_model",
        type=str,
        required=True,
        help="Path to the best model checkpoint",
    )
    parser.add_argument(
        "--huggingface_cache",
        type=str,
        required=True,
        help="Path to Huggingface cache directory",
    )

    # Optional arguments with default values
    parser.add_argument(
        "--normalize_embeddings",
        type=str,
        default="False",
        help="Whether to normalize embeddings (True/False)",
    )
    parser.add_argument(
        "--embeddings_aggr",
        type=str,
        default="mean",
        help="Embeddings aggregation method",
    )
    parser.add_argument(
        "--h_out", type=int, default=768, help="Output hidden dimension"
    )
    parser.add_argument("--h_in", type=int, default=768, help="Input hidden dimension")
    parser.add_argument(
        "--num_proc", type=int, default=1, help="Number of processes to use"
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use this flag to disable CUDA and run on CPU (default: False)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for sampling",
    )
    parser.add_argument(
        "--subsample", type=float, default=1.0, help="Subsampling ratio for data"
    )

    parser.add_argument(
        "--has_contrastive",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Whether to load a model trained with contrastive loss.",
    )
    # Parse arguments
    opts = parser.parse_args()

    # Convert string boolean to actual boolean
    opts.normalize_embeddings = opts.normalize_embeddings.lower() == "true"

    # Validate paths
    required_paths = [
        "path_test_data",
        "logging_path",
        "path_prompt_template",
        "huggingface_cache",
        "best_model",
    ]

    for path_arg in required_paths:
        path = getattr(opts, path_arg)
        if not os.path.exists(path):
            parser.error(f"The path specified by {path_arg} does not exist: {path}")

    return vars(opts)


def run(
    experiment_id,
    path_test_data,
    **kwargs,
):
    # 0. Set the experiment seed
    set_seeds(kwargs["seed"])
    # 1. Initialize run logger
    logger = get_logger(log_level=kwargs["logger_verbosity"])
    logger.info(
        format_with_hashes(
            text=f"Beginning run with seed {kwargs['seed']} and ID {experiment_id}"
        )
    )
    # 2. Load the dataset
    logger.info("\tLoad the testing dataset...")
    definitions_list = read_json(path_json=path_test_data)
    # 3. Shuffle and consider a subset
    random.shuffle(definitions_list)
    n_samples = int(len(definitions_list) * kwargs["subsample"])
    definitions_list = definitions_list[:n_samples]
    # 4. Load the model
    logger.info("\tCreating the model...")
    device = "cpu" if kwargs["use_cpu"] else "cuda"
    if kwargs["has_contrastive"]:
        rule_constrainer = ContrastiveConstrainerModel(
            h_out=kwargs["h_out"],
            h_in=kwargs["h_in"],
            require_encoder=True,  # suppose we always get raw data
            pretrained_model=kwargs["pretrained_model"],
            huggingface_cache=kwargs["huggingface_cache"],
            aggregation=kwargs["embeddings_aggr"],
            normalize=kwargs["normalize_embeddings"],
            temperature=kwargs["temperature"],
        )
    else:
        rule_constrainer = NaiveConstrainerModel(
            pretrained_model=kwargs["pretrained_model"],
            huggingface_cache=kwargs["huggingface_cache"],
            aggregation=kwargs["embeddings_aggr"],
            normalize=kwargs["normalize_embeddings"],
            cache_dir=kwargs["cache_dir"],
        )
    rule_constrainer.load_best(path_best_model=kwargs["best_model"])
    rule_constrainer = rule_constrainer.to(device)
    rule_constrainer.eval()
    # 5. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=kwargs["pretrained_model"],
        cache_dir=kwargs["huggingface_cache"],
    )
    # 6. Iterate over samples
    results = []

    for definition_sample in tqdm(
        definitions_list, desc="Checking matching rules-definitions..."
    ):
        rules = definition_sample["concept"]
        tokenized_rules = tokenize_data(
            tokenizer,
            kwargs["path_prompt_template"],
            rules,
            input_type="rules",
        ).to(device)
        # Add batch dimension
        tokenized_rules = tokenized_rules.unsqueeze(0)
        padding_mask = torch.ones(
            tokenized_rules.shape[0],
            tokenized_rules.shape[1],
            dtype=torch.bool,
            device=device,
        )
        # Get rule embeddings
        with torch.no_grad():
            rule_embeddings = rule_constrainer.encode_rules(
                tokenized_rules, padding_mask
            ).cpu()
        ####### Now, the definitions
        definitions = [definition_sample["true_definition"]] + definition_sample[
            "negative_definitions"
        ]
        tokenized_sentences = tokenize_data(
            tokenizer,
            kwargs["path_prompt_template"],
            definitions,
            input_type="sentences",
        ).to(device)
        # Add batch dimension
        tokenized_sentences = tokenized_sentences.unsqueeze(0)
        # Get sentence embeddings
        with torch.no_grad():
            sentence_embeddings = rule_constrainer.encode_sentences(
                tokenized_sentences
            ).cpu()
        sentence_embeddings = sentence_embeddings.squeeze(0)
        # Now, compute the distances between the sentence embeddings and the rule embeddings
        distances = rule_constrainer.compute_tensor_similarity(
            sentence_embeddings, rule_embeddings
        )
        labels = torch.zeros(distances.shape)
        labels[0] = 1
        results.append(
            {
                "concept": rules,
                "true_definition": definition_sample["true_definition"],
                "MRIs": compute_mri(similarities=distances.T, labels=labels.T),
                "n_rules": len(rules),
            }
        )
    mri_scores = {}
    for result in results:
        mris = result["MRIs"]
        for top_m_value in [1, 3, 5, 10]:
            if top_m_value not in mri_scores:
                mri_scores[top_m_value] = []
            mri_scores[top_m_value].append(mris[f"MRI@{top_m_value}"])

    logger.info("\tReporting the MRIs...")
    final_stats = {}
    for key, values in mri_scores.items():
        final_stats[key] = np.mean(values)
        logger.info("\t\tAvg %s: %s", key, round(final_stats[key], 3))
    final_stats["n_definitions"] = len(results)
    final_stats["avg_rules_x_definition"] = np.mean([el["n_rules"] for el in results])
    final_stats["std_rules_x_definition"] = np.std([el["n_rules"] for el in results])
    logger.info("\tExporting the results...")
    with open(
        os.path.join(kwargs["logging_path"], "predictions.json"), "w+", encoding="utf-8"
    ) as f:
        json.dump(results, f)
    with open(
        os.path.join(kwargs["logging_path"], "metrics.json"), "w+", encoding="utf-8"
    ) as f:
        json.dump(final_stats, f)


if __name__ == "__main__":
    # Get GPU ID from environment variable
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_id is None:
        print("Warning: CUDA_VISIBLE_DEVICES not set")

    # Parse command line arguments
    args = parse_arguments()
    run(**args)
