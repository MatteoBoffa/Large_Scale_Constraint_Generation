import argparse


def get_extraction_parameters(args=None):
    """Functions that gets the parameters passed from the .sh script.
    Args:
        args (_type_): Parameters from the .sh script.
    Returns:
        Dictionary: Parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="Parser to pre-process the a dataset containing, for each row, a rule and one positive example.\
                    We expect 3 partitions: one for training, one for validation, one for test."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="GEM/common_gen",
        help="Path or huggingface link toward the dataset. If path, \
                it must point to the position in which we saved the dataset dict. Default to `GEM/common_gen`.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/huggingface/datasets",
        help="Cache-path where to save the dataset downloaded from huggingface. Default to `~/.cache/huggingface/datasets`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output dir where to save the processed dataset.",
    )
    parser.add_argument(
        "--training_partition",
        type=str,
        default="train",
        help="Training partition. Default to `train`.",
    )
    parser.add_argument(
        "--validation_partition",
        type=str,
        default="validation",
        help="Validation partition. Default to `validation`.",
    )
    parser.add_argument(
        "--testing_partition",
        type=str,
        default="test",
        help="Testing partition. Default to `test`.",
    )
    parser.add_argument(
        "--rules_column",
        type=str,
        required=True,
        help="The dataset column-name containing the list of rules we want to follow for each sample.",
    )
    parser.add_argument(
        "--sample_sentence",
        type=str,
        required=True,
        help="The dataset column-name containing the sample session following the given rules.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        help="Seed to have reproducible experiments when randomness is involved.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Partition of the training set to use for validation. Default to 0.2.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of splits for the K-Fold validation. Default to 5.",
    )

    args = parser.parse_args()
    return args
