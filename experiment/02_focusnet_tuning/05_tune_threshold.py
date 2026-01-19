from itertools import islice
from tqdm import tqdm
from core.common.utils import format_with_hashes, get_logger, set_seeds
from core.contrastive_encoder.classes.rule_encoder import (
    ContrastiveConstrainerModel,
    NaiveConstrainerModel,
)
from core.contrastive_encoder.functions.metrics import (
    calculate_metrics,
    log_scores_distributions,
)
from core.contrastive_encoder.functions.option_parser import get_testing_options
from core.contrastive_encoder.classes.inference_dataset import (
    InferenceConstrainerDataset,
)
from core.rule_checker.random_forest import RandomForestChecker
from core.rule_checker.knn_checker import KNNChecker
from core.rule_checker.logreg_checker import LogRegChecker


def test_loop(
    experiment_id,
    path_test_data,
    macro_rule_id_column,
    **kwargs,
):
    # 0. Set the experiment seed
    set_seeds(kwargs["seed"])
    # 1. Initialize run logger
    logger = get_logger(log_level=kwargs["logger_verbosity"])
    logger.info(
        format_with_hashes(
            text=f"Tuning the classification threshold for model with ID {experiment_id}"
        )
    )
    logger.info("PHASE I - DATASET CREATION")
    # 2. Create the validation dataset
    logger.info("\tCreating the validation dataset...")
    test_dataset = InferenceConstrainerDataset(
        logger=logger,
        paths_data=path_test_data,
        macro_rule_id=macro_rule_id_column,
        pretrained_model=kwargs["pretrained_model"],
        rule_column=kwargs["rule_column"],
        input_column=kwargs["input_column"],
        path_prompt_template=kwargs["path_prompt_template"],
        huggingface_cache=kwargs["huggingface_cache"],
        paths_embeddings=kwargs["path_testing_embeddings"],
        pos_sample_column=kwargs["pos_sample_column"],
        seed=kwargs["seed"],
        partition="test",
        cache_dir=kwargs["cache_dir"],
        prob_neg_sample=kwargs["prob_neg_sample"],
    )
    logger.info("PHASE II - ESTIMATING PROBABILITY THRESHOLD...")
    device = "cpu" if kwargs["use_cpu"] else "cuda"
    # 3. Load the model
    logger.info("\tCreating the model...")
    if kwargs["classifier"] == "nlp_baseline":
        rule_constrainer = None
    elif kwargs["has_contrastive"]:
        rule_constrainer = ContrastiveConstrainerModel(
            h_out=kwargs["h_out"],
            h_in=kwargs["h_in"],
            require_encoder=kwargs["require_encoder"],
            pretrained_model=kwargs["pretrained_model"],
            huggingface_cache=kwargs["huggingface_cache"],
            aggregation=kwargs["embeddings_aggr"],
            normalize=kwargs["normalize_embeddings"],
            temperature=kwargs["temperature"],
        )
        # 4. Get the best model
        rule_constrainer.load_best(path_best_model=kwargs["best_model"])
    else:
        rule_constrainer = NaiveConstrainerModel(
            pretrained_model=kwargs["pretrained_model"],
            huggingface_cache=kwargs["huggingface_cache"],
            aggregation=kwargs["embeddings_aggr"],
            normalize=kwargs["normalize_embeddings"],
            cache_dir=kwargs["cache_dir"],
        )

    # 5. Now, load the rule checker
    logger.info("\tLoading the rule checker...")
    if kwargs["classifier"] == "knn":
        shallow_rule_checker = KNNChecker.load(
            path=kwargs["best_model"], rule_embedder=rule_constrainer, device=device
        )
    elif kwargs["classifier"] == "rf":
        shallow_rule_checker = RandomForestChecker.load(
            path=kwargs["best_model"], rule_embedder=rule_constrainer, device=device
        )
    else:
        shallow_rule_checker = LogRegChecker.load(
            path=kwargs["best_model"], rule_embedder=rule_constrainer, device=device
        )

    logger.info("\tProcessing the validation data and starting the test loop!")
    rules_id, rules, sentences, macro_rule_labels, _, hash_sentences = (
        test_dataset.get_rules_and_sentence()
    )
    # Subsample the data
    subsample = list(
        islice(
            zip(
                rules_id,
                rules,
                sentences,
                hash_sentences,
            ),
            kwargs["subsample"],
        )
    )
    # Prepare the lists to handle the outputs
    scores = []
    # To speed up the process, we will batch the inference
    batch_sentences, batch_rules = [], []
    progress_bar = tqdm(
        subsample, desc="Obtaining scores x sample...", total=len(subsample)
    )
    for _, rule, sentence, _ in subsample:
        # Accumulate batches
        batch_sentences.append(sentence)
        batch_rules.append(rule)
        if len(batch_rules) == kwargs["batch_size"]:
            _, batched_score = shallow_rule_checker.check_compliance(
                sentences=batch_sentences,
                rules=batch_rules,
                device=device,
            )
            batch_sentences, batch_rules = [], []
            scores.extend(batched_score)
            progress_bar.update(len(batched_score))

    # Handle remaining batches
    if batch_rules:
        _, batched_score = shallow_rule_checker.check_compliance(
            sentences=batch_sentences,
            rules=batch_rules,
            device=device,
        )
        scores.extend(batched_score)
        progress_bar.update(len(batched_score))

    progress_bar.close()
    sublabels = macro_rule_labels[: len(subsample)]
    # Now, find the optimal confidence score
    best_threshold, best_precision, best_recall = (
        shallow_rule_checker.find_optimal_confidence_score(
            y_true=sublabels, p=scores, min_precision=0.85
        )
    )

    logger.info("Threshold of: %.2f", best_threshold)
    logger.info("Best Precision: %.2f", best_precision)
    logger.info("Best Recall:  %.2f", best_recall)

    shallow_rule_checker.store_confidence_threshold(
        best_threshold, path=kwargs["best_model"]
    )

    logger.info(format_with_hashes(text="Tuning ended!"))


if __name__ == "__main__":
    opts = get_testing_options()
    test_loop(
        **opts,
    )
