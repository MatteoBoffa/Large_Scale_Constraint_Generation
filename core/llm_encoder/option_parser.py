import argparse


def get_encoding_options(args=None):
    """Functions that gets the parameters passed from the .sh script.
    Args:
        args (_type_): Parameters from the .sh script.
    Returns:
        Dictionary: Parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="Train a rule-encoder using the CommonGEN dataset."
    )
    ################################ Experiments Metadata
    parser.add_argument(
        "--logger_verbosity",
        type=str,
        default="info",
        choices=["debug", "info", "warning"],
        help="The logger verbosity.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=30,
        help="Number of CPU processes. Default to 30.",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use this flag to disable CUDA and run on CPU (default: False)",
    )
    ################################ I/O Parameters
    parser.add_argument(
        "--file_ids",
        type=str,
        nargs="+",
        help="A list containing the ids of the files to encode.",
    )
    parser.add_argument(
        "--file_paths",
        type=str,
        nargs="+",
        help="Paths toward the files to encode. Expecting to point toward `.parquet filesß`.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        required=True,
        help="Path to the file containing the prompt templates to create the rules.",
    )
    parser.add_argument(
        "--rule_column",
        type=str,
        required=True,
        help="Column name where the rules are stored.",
    )
    parser.add_argument(
        "--input_column",
        type=str,
        required=True,
        help="Column name with a sentence respecting the corresponding rule.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where we will store the trained model, the training checkpoints, etc.",
    )
    parser.add_argument(
        "--huggingface_cache",
        type=str,
        default=None,
        help="Path where we save the huggingface cache. If None, we will use the default",
    )
    ################################ Model parameters
    parser.add_argument(
        "--sentence_encoder",
        type=str,
        required=True,
        help="Name/path of the sentence encoder we will use to create the embeddings of the input sentences.",
    )
    parser.add_argument(
        "--center_rule_box_encoder",
        type=str,
        default=None,
        help="Name/path of the sentence encoder we will use to map the rule into the centers of our boxes. If not specified, same as `sentence_encoder`.",
    )
    parser.add_argument(
        "--offset_rule_box_encoder",
        type=str,
        default=None,
        help="Name/path of the sentence encoder we will use to map the rule into the offset of our boxes. If not specified, same as `sentence_encoder`.",
    )
    parser.add_argument(
        "--embeddings_aggr",
        type=str,
        choices=["cls", "mean_pooling"],
        default="mean_pooling",
        help="Type of aggregation to get the sentence embeddings.",
    )
    parser.add_argument(
        "--normalize_embeddings",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Whether to normalize the embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size. Default to 32.",
    )
    ################################
    args = parser.parse_args()
    args.normalize_embeddings = True if args.normalize_embeddings == "True" else False
    # return a dictionary of parameters
    return vars(args)
