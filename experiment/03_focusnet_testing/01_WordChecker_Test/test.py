from itertools import islice
import json
import os
import pandas as pd
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
from core.rule_checker.logreg_checker import LogRegChecker
from core.rule_checker.random_forest import RandomForestChecker
from core.rule_checker.knn_checker import KNNChecker
from core.rule_checker.baseline_NLP_checker import BaselineNLPChecker


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
            text=f"Beginning the contrastive-constrainer testing with seed {kwargs['seed']} and ID {experiment_id}"
        )
    )
    logger.info("PHASE I - DATASET CREATION")
    # 2. Create the testing dataset
    logger.info("\tCreating the testing dataset...")
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
    logger.info("PHASE II - TESTING")
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
        best_threshold = shallow_rule_checker.get_confidence_threshold()
    elif kwargs["classifier"] == "rf":
        shallow_rule_checker = RandomForestChecker.load(
            path=kwargs["best_model"], rule_embedder=rule_constrainer, device=device
        )
        best_threshold = shallow_rule_checker.get_confidence_threshold()
    elif kwargs["classifier"] == "logreg":
        shallow_rule_checker = LogRegChecker.load(
            path=kwargs["best_model"], rule_embedder=rule_constrainer, device=device
        )
        best_threshold = shallow_rule_checker.get_confidence_threshold()
    else:
        baseline_rule_checker = BaselineNLPChecker(
            nltk_data_dir=kwargs["nltk_data_dir"]
        )
        best_threshold = 0

    logger.info("\tProcessing the testing data and starting the test loop!")
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
    compliance_lists, scores = [], []
    # Also, we want to save the real rules and sentences
    real_rules, real_sentences = [], []
    # To speed up the process, we will batch the inference
    batch_sentences, batch_rules = [], []
    progress_bar = tqdm(
        subsample,
        desc=f"Checking testing compliances (using threshold: {best_threshold:2f})...",
        total=len(subsample),
    )

    for rule_id, rule, sentence, hash_sentence in subsample:
        # Get the real rule and sentence
        real_rule = test_dataset.gather_real_rule(hash_rules=rule_id)
        real_sentence = test_dataset.gather_real_sentences(hash_sentence=hash_sentence)
        real_rules.append(real_rule)
        real_sentences.append(real_sentence)
        # If we are using the NLP baseline, we do not need batching
        if kwargs["classifier"] == "nlp_baseline":
            compliance, score = baseline_rule_checker.check_compliance(
                sentence=real_sentence, rules=real_rule
            )
            compliance_lists.append(compliance)
            scores.append(score)
            progress_bar.update(1)
        else:
            # Accumulate batches
            batch_sentences.append(sentence)
            batch_rules.append(rule)
            if len(batch_rules) == kwargs["batch_size"]:
                batched_compliances, batched_score = (
                    shallow_rule_checker.check_compliance(
                        sentences=batch_sentences,
                        rules=batch_rules,
                        confidence_score=best_threshold,
                        device=device,
                    )
                )
                batch_sentences, batch_rules = [], []
                compliance_lists.extend(batched_compliances)
                scores.extend(batched_score)
                progress_bar.update(len(batched_compliances))

    # Handle remaining batches
    if batch_rules:
        batched_compliances, batched_score = shallow_rule_checker.check_compliance(
            sentences=batch_sentences,
            rules=batch_rules,
            confidence_score=best_threshold,
            device=device,
        )
        compliance_lists.extend(batched_compliances)
        scores.extend(batched_score)
        progress_bar.update(len(batched_compliances))

    progress_bar.close()
    sublabels = macro_rule_labels[: len(subsample)]
    # Calculate metrics
    macro_rule_metrics = calculate_metrics(sublabels, compliance_lists)
    # Save the results
    output_df = pd.DataFrame(
        zip(
            real_rules,
            real_sentences,
            sublabels,
            compliance_lists,
            scores,
        ),
        columns=[
            "rule",
            "sentence",
            "label",
            "macro_prediction",
            "score",
        ],
    )
    output_df.to_json(
        os.path.join(kwargs["logging_path"], "labels_predictions.json"),
        orient="index",
        indent=4,
    )
    # 7. Logging the metrics
    logger.info("Macro rules accuracy score: %.3f", macro_rule_metrics["accuracy"])
    with open(
        os.path.join(kwargs["logging_path"], "metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "macro_rules": macro_rule_metrics,
            },
            f,
            indent=4,
        )
    # 8. Log the LOF score distributions
    figure_correct, figure_incorrect = log_scores_distributions(
        y_true=sublabels,
        y_pred=compliance_lists,
        lof_scores=scores,
    )
    figure_correct.savefig(os.path.join(kwargs["logging_path"], "correct_scores.png"))
    figure_incorrect.savefig(
        os.path.join(kwargs["logging_path"], "incorrect_scores.png")
    )
    # 10. Exit
    logger.info(format_with_hashes(text="Inference ended!"))


if __name__ == "__main__":
    opts = get_testing_options()
    test_loop(
        **opts,
    )
