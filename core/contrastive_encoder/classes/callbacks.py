import json
import logging
import os
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class ControlSetEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        control_dataloader: DataLoader,
        tensorboard_writer: SummaryWriter = None,
        logger: logging.Logger = None,
        metric_fn=None,
        device="cpu",
        top_m_values=(1, 3, 5, 10),
        show_progress=True,
        eval_frequency: int = 1,
    ):
        super().__init__()
        self.control_dataloader = control_dataloader
        self.tensorboard_writer = tensorboard_writer
        self.logger = logger
        self.metric_fn = metric_fn
        self.device = device
        self.top_m_values = top_m_values
        self.show_progress = show_progress
        self.eval_frequency = eval_frequency
        self.best_metric = -10_000  # random negative value (MRI is a positive value)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Add control metrics while preserving validation metrics"""
        if metrics is None:
            metrics = {}

        # Store the original validation loss
        eval_loss = metrics.get("eval_loss", None)

        model = kwargs.get("model")
        if model is None:
            return control

        model.eval()

        control_metrics = model.extract_MRI_metrics(
            dataloader=self.control_dataloader,
            show_progress_bar=self.show_progress,
            desc="Evaluating control set",
            top_m_values=self.top_m_values,
            device=self.device,
        )

        # Add MRI@1 as a primary metric for model selection
        metrics["eval_mri1"] = control_metrics["control/MRI@1"]

        # Preserve the validation loss if it exists
        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss

        # Add control metrics
        metrics.update(control_metrics)

        if state.is_local_process_zero:
            if self.logger is not None:
                result_string = " | ".join(
                    [f"eval_loss: {eval_loss:.4f}" if eval_loss is not None else ""]
                    + [
                        f"{metric_name.replace('control/', '')}: {value:.4f}"
                        for metric_name, value in control_metrics.items()
                    ]
                )
                s = f"\nEvaluation - Epoch {state.epoch}: {result_string}"
                self.logger.info(s)

            if self.tensorboard_writer is not None:
                for metric_name, value in control_metrics.items():
                    self.tensorboard_writer.add_scalar(
                        metric_name, value, state.global_step
                    )
                if eval_loss is not None:
                    self.tensorboard_writer.add_scalar(
                        "eval/loss", eval_loss, state.global_step
                    )
                self.tensorboard_writer.flush()
        # Save the best metric
        if metrics.get("eval_mri1", 0) >= self.best_metric:
            self.best_metric = metrics["eval_mri1"]  # Update the best metric
            state.best_metric = self.best_metric  # Update the state metric
            best_epoch_info = {"epoch": state.epoch, "eval_mri1": metrics["eval_mri1"]}
            best_folder_model = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_folder_model, exist_ok=True)
            with open(
                os.path.join(best_folder_model, "best_epoch_info.json"),
                "w+",
                encoding="utf-8",
            ) as f:
                json.dump(best_epoch_info, f)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Keep the existing epoch end behavior"""
        if (state.epoch + 1) % self.eval_frequency != 0:
            return control
        # The evaluation will be handled by on_evaluate
        return control
