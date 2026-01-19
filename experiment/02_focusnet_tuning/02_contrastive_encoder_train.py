import os
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
from core.common.utils import format_with_hashes, get_logger, set_seeds
from core.contrastive_encoder.classes.callbacks import (
    ControlSetEvaluationCallback,
)
from core.contrastive_encoder.classes.rule_encoder import ContrastiveConstrainerModel
from core.contrastive_encoder.functions.metrics import (
    compute_training_similarity_metrics,
)
from core.contrastive_encoder.functions.option_parser import get_training_options
from core.contrastive_encoder.classes.training_dataset import (
    TrainingContrastiveConstrainerDataset,
)


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
    train_dataset = TrainingContrastiveConstrainerDataset(
        logger=logger,
        paths_data=path_training_data,
        macro_rule_id=macro_rule_id_column,
        pretrained_model=kwargs["pretrained_model"],
        rule_column=kwargs["rule_column"],
        input_column=kwargs["input_column"],
        path_prompt_template=kwargs["path_prompt_template"],
        huggingface_cache=kwargs["huggingface_cache"],
        paths_embeddings=kwargs["path_training_embeddings"],
        pos_sample_column=pos_sample_column,
        seed=kwargs["seed"],
        partition="train",
        subsample=kwargs["subsample"],
        cache_dir=kwargs["cache_dir"],
        n_negative_samplings=kwargs["negative_samples"],
    )
    logger.info("\tCreating the validation dataset (might take a while)...")
    validation_dataset = TrainingContrastiveConstrainerDataset(
        logger=logger,
        paths_data=path_validation_data,
        macro_rule_id=macro_rule_id_column,
        pretrained_model=kwargs["pretrained_model"],
        rule_column=kwargs["rule_column"],
        input_column=kwargs["input_column"],
        path_prompt_template=kwargs["path_prompt_template"],
        huggingface_cache=kwargs["huggingface_cache"],
        paths_embeddings=kwargs["path_validation_embeddings"],
        pos_sample_column=pos_sample_column,
        seed=kwargs["seed"],
        partition="validation",
        subsample=kwargs["subsample"],
        cache_dir=kwargs["cache_dir"],
        n_negative_samplings=kwargs["negative_samples"],
    )
    logger.info(
        "\tNow, creating a control dataset in which we will calculate the MRI..."
    )
    control_dataset = TrainingContrastiveConstrainerDataset(
        logger=logger,
        paths_data=path_validation_data,
        macro_rule_id=macro_rule_id_column,
        pretrained_model=kwargs["pretrained_model"],
        rule_column=kwargs["rule_column"],
        input_column=kwargs["input_column"],
        huggingface_cache=kwargs["huggingface_cache"],
        path_prompt_template=kwargs["path_prompt_template"],
        paths_embeddings=kwargs["path_validation_embeddings"],
        pos_sample_column=pos_sample_column,
        seed=kwargs["seed"],
        partition="control",
        subsample=kwargs["subsample"],
        cache_dir=kwargs["cache_dir"],
        n_negative_samplings=30,
    )
    # Create a Dataloader for the control dataset
    control_dataloader = DataLoader(
        control_dataset, batch_size=kwargs["batch_size"], shuffle=False, drop_last=True
    )

    device = "cpu" if kwargs["use_cpu"] else "cuda"
    writer = SummaryWriter(kwargs["logging_path"])
    control_callback = ControlSetEvaluationCallback(
        control_dataloader=control_dataloader,
        tensorboard_writer=writer,
        logger=logger,
        show_progress=True,
        eval_frequency=1,
        device=device,
    )

    logger.info("PHASE II - TRAINING")
    # 5. Training
    # training_steps_x_epoch = ceil(len(train_dataset) / kwargs["batch_size"])
    training_arguments = TrainingArguments(
        run_name=experiment_id,
        output_dir=os.path.join(output_path, "training_result"),
        logging_dir=kwargs["logging_path"],
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        eval_on_start=False,
        logging_first_step=False,
        per_device_train_batch_size=kwargs["batch_size"],
        per_device_eval_batch_size=kwargs["batch_size"],
        learning_rate=kwargs["lr"],
        lr_scheduler_type=kwargs["lr_scheduler"],
        num_train_epochs=kwargs["epochs"],
        save_strategy="epoch",
        use_cpu=kwargs["use_cpu"],
        seed=kwargs["seed"],
        dataloader_num_workers=kwargs["num_proc"],
        dataloader_prefetch_factor=10,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_total_limit=1,
        warmup_steps=100,  # Add warmup steps
        weight_decay=0.01,  # Add some regularization
        report_to=["tensorboard"],  # This is crucial - disable default logging
        metric_for_best_model="loss",
        # greater_is_better=True,  # MRI should be maximized
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
        eval_dataset=validation_dataset,
        compute_metrics=compute_training_similarity_metrics,
        callbacks=[control_callback],
    )
    # 8. Train
    logger.info("\tStart the training!")
    try:
        trainer.train()
    finally:
        # Make sure to close the writer
        writer.close()
    # 9. Exit without saving the best model in this phase > we are only interested in the MRI
    logger.info(format_with_hashes(text="Training ended!"))


if __name__ == "__main__":
    opts = get_training_options()
    training_loop(
        **opts,
    )
