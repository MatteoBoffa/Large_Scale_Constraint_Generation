from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput
import torch
import warnings


class RuleConstrainerTrainer(Trainer):
    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        """
        Compute the training loss for the model.

        Args:
            model: The model to train
            inputs: The inputs to the model
            num_items_in_batch: Optional batch size
            return_outputs: Whether to return the model outputs along with the loss

        Returns:
            loss or tuple of (loss, outputs)
        """
        # Forward pass through the model
        outputs = model(**inputs)

        # Get the loss from the outputs dictionary
        loss = outputs["loss"]

        # Handle potential invalid loss values
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(f"Invalid loss value detected in trainer: {loss}")
            loss = torch.tensor(0.0, device=self.args.device)

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self, dataloader, description: str, prediction_loss_only: bool = None, **kwargs
    ):
        """
        Evaluation loop for validation and testing.

        Args:
            dataloader: The dataloader for evaluation
            description: Description of the evaluation phase
            prediction_loss_only: Whether to only compute the loss
            **kwargs: Additional arguments

        Returns:
            EvalLoopOutput containing evaluation results
        """
        # Get metrics from the model's evaluation loop
        model_metrics = self.model.evaluation_loop(
            dataloader=dataloader, device=self.args.device
        )

        # Create and return the evaluation output
        eval_output = EvalLoopOutput(
            predictions=None,  # No predictions needed as per the implementation
            label_ids=None,  # No label IDs needed
            metrics=model_metrics,  # Include the model's custom metrics
            num_samples=len(dataloader.dataset),  # Total number of evaluated samples
        )

        return eval_output
