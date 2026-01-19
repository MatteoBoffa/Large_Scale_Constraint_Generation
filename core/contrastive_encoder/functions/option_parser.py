import argparse


def get_training_options(args=None):
    """Functions that gets the parameters passed from the .sh script.
    Args:
        args (_type_): Parameters from the .sh script.
    Returns:
        Dictionary: Parsed parameters.
    """
    parser = argparse.ArgumentParser(description="Parameters to train a rule-encoder.")
    ################################ Experiments Metadata
    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="The experiment identifier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        help="The experiment seed.",
    )
    parser.add_argument(
        "--logger_verbosity",
        type=str,
        default="info",
        choices=["debug", "info", "warning"],
        help="Log verbosity. Can be debug, info or warning.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=10,
        help="Number of CPU processes. Default to 10.",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use this flag to disable CUDA and run on CPU (default: False)",
    )
    ################################ I/O Parameters
    parser.add_argument(
        "--path_training_data",
        type=str,
        required=True,
        help="Path to the training data. Expecting a parquet file.",
    )
    parser.add_argument(
        "--path_training_embeddings",
        type=str,
        default=None,
        help="Path to the training embeddings. If `None`, we will use an `pretrained_model` to embed the text.",
    )
    parser.add_argument(
        "--path_validation_data",
        type=str,
        required=True,
        help="Path to the validation data. Expecting a parquet file.",
    )
    parser.add_argument(
        "--path_validation_embeddings",
        type=str,
        default=None,
        help="Path to the validation embeddings. If `None`, we will use an `pretrained_model` to embed the text.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Whether we sub-sample the initial datasets to a `subsample` number of rows. Default to None.",
    )
    parser.add_argument(
        "--pos_sample_column",
        type=str,
        required=True,
        help="Column name to identify rows respecting the same rule.",
    )
    parser.add_argument(
        "--macro_rule_id_column",
        type=str,
        required=True,
        help="Column name to identify rows corresponding to the same macro rule (e.g. you must respect the following rules: r1, r2, ...).",
    )
    parser.add_argument(
        "--rule_column",
        type=str,
        default=None,
        help="Column name containing the rules. Unnecessary if the embeddings were precomputed.",
    )
    parser.add_argument(
        "--input_column",
        type=str,
        default=None,
        help="Column name containing the positive samples per rules. Unnecessary if the embeddings were precomputed.",
    )
    parser.add_argument(
        "--path_prompt_template",
        type=str,
        default=None,
        help="Path to json file containing the prompts to pre-append to the rules and input columns.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Name of the huggingface model we will use to embed the rules and sentences, if the embeddings are not provided yet.",
    )
    parser.add_argument(
        "--huggingface_cache",
        type=str,
        default=None,
        help="Path where we save the huggingface cache. If None, we will use the default",
    )
    parser.add_argument(
        "--embeddings_aggr",
        type=str,
        choices=["cls", "mean_pooling"],
        default="mean_pooling",
        help="Type of aggregation to get the sentence embeddings. Useful if the embeddings are not cached.",
    )
    parser.add_argument(
        "--normalize_embeddings",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Whether to normalize the embeddings. Useful if the embeddings are not cached.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where we will store the trained model, the training checkpoints, etc.",
    )
    parser.add_argument(
        "--logging_path",
        type=str,
        default=None,
        help="Path where we will store the training logs.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path where we will cache all the intermidiate pre-processed dataset, given an `experiment_id` and `seed`.",
    )
    ################################ Training model parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of training epochs. Default to 250.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate. Default to 1e-4.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Mini-batch size. Default to 512.",
    )
    parser.add_argument(
        "--n_evals_x_epoch",
        type=int,
        default=1,
        help="How many evaluation steps to perform within an epoch. Default to 1.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        choices=["linear", "reduce_lr_on_plateau"],
        help="LR scheduler. Can be `linear` or `reduce_lr_on_plateau`. Default to `linear`.",
    )
    parser.add_argument(
        "--h_in",
        type=int,
        default=768,
        help="Hidden size of rules and sentence embeddings. For now, assumed to be the same, Default to 768 (output size of basic sentence encoders).",
    )
    parser.add_argument(
        "--h_out",
        type=int,
        default=768,
        help="Hidden size of the rules constrainer space. Default to 768 (same as `h_in`)",
    )
    parser.add_argument(
        "--negative_samples",
        type=int,
        default=2,
        help="Number of negative samples we want to force per batch. Default to 2.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for the contrastive loss. Degault to 0.1.",
    )
    parser.add_argument(
        "--loss_rules",
        type=str,
        default="False",
        help="Whether to also compute a loss for the rules (default: False)",
    )

    parser.add_argument(
        "--classifier",
        type=str,
        choices=["rf", "knn", "logreg"],
        default="rf",
        help=(
            "Classification head to use for rule compliance checking. "
            "Options: 'rf' (Random Forest), 'knn' (K-Nearest Neighbors) or 'logreg' (Logistic Regression)"
            "Default: rf"
        ),
    )

    ################################
    args = parser.parse_args()
    args.loss_rules = True if args.loss_rules == "True" else False
    args.require_encoder = False if args.path_training_embeddings else True
    args.normalize_embeddings = True if args.normalize_embeddings == "True" else False
    # return a dictionary of parameters
    return vars(args)


def get_testing_options(args=None):
    """Functions that gets the parameters passed from the .sh script.
    Args:
        args (_type_): Parameters from the .sh script.
    Returns:
        Dictionary: Parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="Parameters to use a trained rule-encoder to get some testing stats."
    )
    ################################ Experiments Metadata
    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="The experiment identifier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        help="The experiment seed.",
    )
    parser.add_argument(
        "--logger_verbosity",
        type=str,
        default="info",
        choices=["debug", "info", "warning"],
        help="Log verbosity. Can be debug, info or warning.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=10,
        help="Number of CPU processes. Default to 10.",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use this flag to disable CUDA and run on CPU (default: False)",
    )
    ################################ I/O Parameters
    parser.add_argument(
        "--path_test_data",
        type=str,
        required=True,
        help="Path to the testing data. Expecting a parquet file.",
    )
    parser.add_argument(
        "--path_testing_embeddings",
        type=str,
        default=None,
        help="Path to the training embeddings. If `None`, we will use an `pretrained_model` to embed the text.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Whether we sub-sample the initial datasets to a `subsample` number of rows. Default to None.",
    )
    parser.add_argument(
        "--pos_sample_column",
        type=str,
        default=None,
        help="Column name to identify rows respecting the same rule. Optional, as it might be unknown at test time.",
    )
    parser.add_argument(
        "--macro_rule_id_column",
        type=str,
        required=True,
        help="Column name to identify rows corresponding to the same macro rule (e.g. you must respect the following rules: r1, r2, ...).",
    )
    parser.add_argument(
        "--rule_column",
        type=str,
        default=None,
        help="Column name containing the rules. Unnecessary if the embeddings were precomputed.",
    )
    parser.add_argument(
        "--input_column",
        type=str,
        default=None,
        help="Column name containing the positive samples per rules. Unnecessary if the embeddings were precomputed.",
    )
    parser.add_argument(
        "--path_prompt_template",
        type=str,
        default=None,
        help="Path to json file containing the prompts to pre-append to the rules and input columns.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Name of the huggingface model we will use to embed the rules and sentences, if the embeddings are not provided yet.",
    )
    parser.add_argument(
        "--huggingface_cache",
        type=str,
        default=None,
        help="Path where we save the huggingface cache. If None, we will use the default",
    )
    parser.add_argument(
        "--embeddings_aggr",
        type=str,
        choices=["cls", "mean_pooling"],
        default="mean_pooling",
        help="Type of aggregation to get the sentence embeddings. Useful if the embeddings are not cached.",
    )
    parser.add_argument(
        "--normalize_embeddings",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Whether to normalize the embeddings. Useful if the embeddings are not cached.",
    )
    parser.add_argument(
        "--logging_path",
        type=str,
        default=None,
        help="Path where we will store the training logs.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path where we will cache all the intermidiate pre-processed dataset, given an `experiment_id` and `seed`.",
    )
    ################################ Parameters for the finetuned model
    parser.add_argument(
        "--best_model",
        type=str,
        required=True,
        help="Path toward the best model. Expecting the directory toward the `.safetensor` file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Mini-batch size. Default to 512.",
    )
    parser.add_argument(
        "--h_in",
        type=int,
        default=768,
        help="Hidden size of rules and sentence embeddings. For now, assumed to be the same, Default to 768 (output size of basic sentence encoders).",
    )
    parser.add_argument(
        "--h_out",
        type=int,
        default=768,
        help="Hidden size of the rules constrainer space. Default to 768 (same as `h_in`)",
    )
    parser.add_argument(
        "--prob_neg_sample",
        type=float,
        default=0.3,
        help="If we are at test phase (e.g., if we know the pos_sample_column), then we can extract some negative examples to see whether the model recognize them. Default probability to 0.3.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for the contrastive loss. Degault to 0.1.",
    )

    parser.add_argument(
        "--has_contrastive",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Whether to load a model trained with contrastive loss.",
    )

    parser.add_argument(
        "--classifier",
        type=str,
        choices=["rf", "knn", "logreg", "nlp_baseline"],
        default="rf",
        help=(
            "Classification head to use for rule compliance checking. "
            "Options: 'rf' (Random Forest), 'knn' (K-Nearest Neighbors), 'logreg' (Logistic Regression), or 'nlp_baseline' (NLP Lemma Baseline). "
            "Default: rf"
        ),
    )

    parser.add_argument(
        "--nltk_data_dir",
        type=str,
        default=None,
        help="Path where we save the NLTK data. If None, we will use the default",
    )

    ################################
    args = parser.parse_args()
    args.require_encoder = False if args.path_testing_embeddings else True
    args.normalize_embeddings = True if args.normalize_embeddings == "True" else False
    args.has_contrastive = True if args.has_contrastive == "True" else False
    # return a dictionary of parameters
    return vars(args)
