import os
from core.common.utils import format_with_hashes, get_logger, set_seeds
from core.contrastive_encoder.classes.rule_encoder import (
    NaiveConstrainerModel,
)
from core.contrastive_encoder.functions.option_parser import get_training_options
from core.contrastive_encoder.classes.training_dataset import (
    TrainingContrastiveConstrainerDataset,
)
from core.rule_checker.random_forest import RandomForestChecker
from core.rule_checker.knn_checker import KNNChecker


def training_loop(
    experiment_id,
    path_training_data,
    path_validation_data,
    pos_sample_column,
    macro_rule_id_column,
    output_path,
    **kwargs,
):
    # 0. Set the experiment seed
    set_seeds(kwargs["seed"])
    # 1. Initialize run logger
    logger = get_logger(log_level=kwargs["logger_verbosity"])
    logger.info(
        format_with_hashes(
            text=f"Beginning the contrastive-constrainer training with ID {experiment_id}"
        )
    )
    logger.info("PHASE I - DATASET CREATION")
    # 2. Create training and validation datasets
    logger.info("\tCreating the training dataset (might take a while)...")
    full_training_paths = [path_training_data, path_validation_data]
    if kwargs["path_training_embeddings"] and kwargs["path_validation_embeddings"]:
        full_embeddings_paths = [
            kwargs["path_training_embeddings"],
            kwargs["path_validation_embeddings"],
        ]
    else:
        full_embeddings_paths = None
    train_dataset = TrainingContrastiveConstrainerDataset(
        logger=logger,
        paths_data=full_training_paths,
        macro_rule_id=macro_rule_id_column,
        pretrained_model=kwargs["pretrained_model"],
        rule_column=kwargs["rule_column"],
        input_column=kwargs["input_column"],
        path_prompt_template=kwargs["path_prompt_template"],
        huggingface_cache=kwargs["huggingface_cache"],
        paths_embeddings=full_embeddings_paths,
        pos_sample_column=pos_sample_column,
        seed=kwargs["seed"],
        partition="train",
        subsample=kwargs["subsample"],
        cache_dir=kwargs["cache_dir"],
        n_negative_samplings=kwargs["negative_samples"],
    )
    device = "cpu" if kwargs["use_cpu"] else "cuda"
    # 3. Load an off-the-shelf encoder model
    logger.info("PHASE II - LOAD AN OFF-THE-SHELF ENCODER MODEL")
    rule_constrainer = NaiveConstrainerModel(
        pretrained_model=kwargs["pretrained_model"],
        huggingface_cache=kwargs["huggingface_cache"],
        aggregation=kwargs["embeddings_aggr"],
        normalize=kwargs["normalize_embeddings"],
        cache_dir=kwargs["cache_dir"],
    ).to(device)
    # 4. Saving it for later model
    best_model_path = os.path.join(output_path, "best_model")
    rule_constrainer.save_safetensors(best_model_path)
    # 4. Now, train a RF Rule Compliance Checker for inference using the training stats
    logger.info("\tNow, fit a classifier with the training data...")
    if kwargs["classifier"] == "rf":
        rule_checker = RandomForestChecker(rule_embedder=rule_constrainer)
    else:
        rule_checker = KNNChecker(rule_embedder=rule_constrainer)
    # 5. Extract rules and sentences from the dataset
    rules, positive_samples = train_dataset.get_rules_and_positive_samples()
    # 6. Train the Random Forest
    rule_checker.fit(
        rules=rules,
        positive_samples=positive_samples,
        device=device,
        percentage_negatives=1,
        batch_size=kwargs["batch_size"],
        n_proc=kwargs["num_proc"],
        logger=logger,
    )
    # 7. Save checker
    rule_checker.save(best_model_path)
    # 8. Exit
    logger.info(format_with_hashes(text="Training ended!"))


if __name__ == "__main__":
    opts = get_training_options()
    training_loop(
        **opts,
    )
