import os
from transformers import TrainingArguments, Trainer
from core.common.utils import format_with_hashes, get_logger, set_seeds
from core.contrastive_encoder.classes.rule_encoder import ContrastiveConstrainerModel
from core.contrastive_encoder.functions.option_parser import get_training_options
from core.contrastive_encoder.classes.training_dataset import (
    TrainingContrastiveConstrainerDataset,
)
from core.rule_checker.random_forest import RandomForestChecker
from core.rule_checker.knn_checker import KNNChecker
from core.rule_checker.logreg_checker import LogRegChecker


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
    logger.info("PHASE II - TRAINING")
    # 5. Training
    # training_steps_x_epoch = ceil(len(train_dataset) / kwargs["batch_size"])
    training_arguments = TrainingArguments(
        run_name=experiment_id,
        output_dir=os.path.join(output_path),
        logging_dir=kwargs["logging_path"],
        do_train=True,
        do_eval=False,
        logging_strategy="epoch",
        logging_first_step=True,
        per_device_train_batch_size=kwargs["batch_size"],
        learning_rate=kwargs["lr"],
        lr_scheduler_type=kwargs["lr_scheduler"],
        num_train_epochs=kwargs["epochs"],
        save_strategy="epoch",
        use_cpu=kwargs["use_cpu"],
        seed=kwargs["seed"],
        dataloader_num_workers=kwargs["num_proc"],
        dataloader_prefetch_factor=10,
        remove_unused_columns=False,
        load_best_model_at_end=False,
        save_total_limit=1,
        report_to=["tensorboard"],  # This is crucial - disable default logging
    )

    # 6. Model
    logger.info("\tCreating the model...")
    rule_constrainer = ContrastiveConstrainerModel(
        h_out=kwargs["h_out"],
        h_in=kwargs["h_in"],
        require_encoder=kwargs["require_encoder"],
        pretrained_model=kwargs["pretrained_model"],
        huggingface_cache=kwargs["huggingface_cache"],
        aggregation=kwargs["embeddings_aggr"],
        normalize=kwargs["normalize_embeddings"],
        temperature=kwargs["temperature"],
        loss_rules=kwargs["loss_rules"],
    ).to(device=device)
    # 7. Trainer
    logger.info("\tCreating the Trainer...")
    trainer = Trainer(
        model=rule_constrainer,
        args=training_arguments,
        train_dataset=train_dataset,
    )
    # 8. Train
    logger.info("\tStart the training!")
    trainer.train()
    # Save the model (it will already be the best one due to load_best_model_at_end=True)
    logger.info("\tSaving the best model...")
    # 9. Saving best model
    best_model_path = os.path.join(output_path, "best_model")
    trainer.save_model(best_model_path)
    # 10. Now, train a LOF-based Rule Compliance Checker for inference using the training stats
    logger.info("\tNow, fit a classifier with the training data...")
    if kwargs["classifier"] == "rf":
        rule_checker = RandomForestChecker(rule_embedder=rule_constrainer)
    elif kwargs["classifier"] == "knn":
        rule_checker = KNNChecker(rule_embedder=rule_constrainer)
    else:
        rule_checker = LogRegChecker(rule_embedder=rule_constrainer)
    # 11. Extract rules and sentences from the dataset
    rules, positive_samples = train_dataset.get_rules_and_positive_samples()
    # 12. Train the Random Forest
    rule_checker.fit(
        rules=rules,
        positive_samples=positive_samples,
        device=device,
        percentage_negatives=1,
        batch_size=kwargs["batch_size"],
        n_proc=kwargs["num_proc"],
        logger=logger,
    )
    # 13. Save checker
    rule_checker.save(os.path.join(output_path, "best_model"))
    # 14. Exit
    logger.info(format_with_hashes(text="Training ended!"))


if __name__ == "__main__":
    opts = get_training_options()
    training_loop(
        **opts,
    )
