import os
import glob
import re
import torch
import warnings
import numpy as np
from typing import Union, List, Tuple
from sklearn.metrics import precision_recall_fscore_support
from tensorboard.backend.event_processing import event_accumulator
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def compute_training_similarity_metrics(eval_pred):
    """
    Custom metric computation function for the HuggingFace Trainer.
    """
    # This contains all the metrics we added in forward()
    sentence_loss = eval_pred.predictions[0]
    if len(eval_pred.predictions) == 3:
        rule_loss = eval_pred.predictions[1]
    else:
        rule_loss = torch.tensor(0.0)
    metrics = eval_pred.predictions[-1]
    # Log all metrics we computed in the forward pass
    return {
        "sentence_loss": sentence_loss.mean().item(),
        "rule_loss": rule_loss.mean().item(),
        "sentence_sparsity": metrics["sentence_sparsity"].mean(),
        "rules_sparsity": metrics["rules_sparsity"].mean(),
        "pos_sent_2_rule_sim_mean": metrics["pos_sent_2_rule_sim_mean"].mean(),
        "neg_sent_2_rule_sim_mean": metrics["neg_sent_2_rule_sim_mean"].mean(),
        "positive_rule_to_rule_sims": metrics["positive_rule_to_rule_sims"].mean(),
        "negative_rule_to_rule_sims": metrics["negative_rule_to_rule_sims"].mean(),
    }


def compute_batch_gini_sparsity(
    tensor: torch.Tensor, threshold: float = 1e-10
) -> torch.Tensor:
    """
    Compute average Gini sparsity across batches efficiently.

    Args:
        tensor: Input tensor of shape (B, H)
        threshold: Values below this are considered zero

    Returns:
        Average Gini coefficient across batches
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor (B,H), got shape {tensor.shape}")

    # Get absolute values for entire batch at once
    x = tensor.abs()

    # Handle zero tensors
    zero_mask = torch.all(x <= threshold, dim=1)

    # Sort all batches at once
    x_sorted = torch.sort(x, dim=1).values
    n = x.size(1)

    # Compute indices once
    index = torch.arange(1, n + 1, device=tensor.device, dtype=torch.float32)

    # Vectorized Gini computation
    numerator = torch.sum((2 * index - n - 1) * x_sorted, dim=1)
    denominator = n * torch.sum(x, dim=1)

    # Handle zero denominator cases
    gini = torch.where(
        denominator > threshold, numerator / denominator, torch.ones_like(numerator)
    )

    return gini.mean()


def compute_mri(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    top_m_values: Union[List[int], Tuple[int, ...]] = (1, 3, 5, 10),
) -> dict:
    """
    Compute Mean Reciprocal Index (MRI) for different top-M values.

    Args:
        similarities: Tensor of shape [B, BxN] containing similarity scores
        labels: Binary tensor of shape [B, BxN] where each row has exactly one 1
        top_m_values: Tuple/List of M values to compute MRI for

    Returns:
        Dictionary containing MRI scores for each M value
    """
    if not torch.is_tensor(similarities):
        similarities = torch.tensor(similarities)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    # Move tensors to the same device if needed
    if similarities.device != labels.device:
        labels = labels.to(similarities.device)

    # Verify shapes match
    assert (
        similarities.shape == labels.shape
    ), f"Shape mismatch: similarities {similarities.shape} vs labels {labels.shape}"

    # Verify each row in labels has exactly one 1
    assert torch.all(
        labels.sum(dim=1) == 1
    ), "Each row in labels must have exactly one 1"

    batch_size = similarities.shape[0]
    metrics = {}

    # Find indices where labels are 1
    target_indices = torch.where(labels == 1)[1]

    # For each row, get the ranking of the target index
    rankings = []
    for i in range(batch_size):
        # Sort similarities in descending order and get indices
        _, sorted_indices = torch.sort(similarities[i], descending=True)
        # Find where the target index appears in the sorted list
        target_pos = (sorted_indices == target_indices[i]).nonzero().item()
        rankings.append(target_pos)

    rankings = torch.tensor(rankings)

    # Compute MRI for each M value
    for M in top_m_values:
        # Check if target is in top M positions (0-based indexing)
        in_top_m = rankings < M
        mri_score = in_top_m.float().mean().item()
        metrics[f"MRI@{M}"] = mri_score

    return metrics


def compute_similarities_metrics(
    similarities: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Compute similarities metrics based on similarity scores:
    - Similarity between target similarity score and its position's score
    - Average similarity between target similarity score and all other scores

    Args:
        similarities: Tensor of shape [B, BxN] containing similarity scores
        labels: Binary tensor of shape [B, BxN] where each row has exactly one 1

    Returns:
        Dictionary containing similarity metrics:
        - target_similarity: Average similarity score distance to target
        - other_similarity: Average similarity score distance to all other positions
    """
    if not torch.is_tensor(similarities):
        similarities = torch.tensor(similarities)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    if similarities.device != labels.device:
        labels = labels.to(similarities.device)

    assert (
        similarities.shape == labels.shape
    ), f"Shape mismatch: similarities {similarities.shape} vs labels {labels.shape}"

    assert torch.all(
        labels.sum(dim=1) == 1
    ), "Each row in labels must have exactly one 1"

    batch_size = similarities.shape[0]

    # Find indices where labels are 1
    target_indices = torch.where(labels == 1)[1]

    target_similarities = []
    other_similarities = []

    for i in range(batch_size):
        # Get the similarity score for the target
        target_similarity = similarities[i, target_indices[i]].item()
        # Get all similarities except the target similarity
        mask = torch.ones(
            similarities[i].shape, dtype=torch.bool, device=similarities.device
        )
        mask[target_indices[i]] = False  # Exclude the target index
        other_avg_similarity = (
            similarities[i][mask].mean().item()
        )  # Apply the mask and get the avg
        target_similarities.append(target_similarity)
        other_similarities.append(other_avg_similarity)

    metrics = {
        "target_similarities": sum(target_similarities) / batch_size,
        "other_similarities_mean": sum(other_similarities) / batch_size,
    }

    return metrics


def calculate_metrics(y_true, y_pred, average="binary"):
    """
    Calculate precision, recall, F1-score, support and accuracy using sklearn.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: String indicating the averaging strategy.
                Can be 'binary' (default), 'micro', 'macro', 'weighted', or None

    Returns:
        dict: Dictionary containing the calculated metrics

    Remember:
    For binary classification (average='binary'):
        Precision: Out of all the instances we predicted as positive (1), how many were actually positive
        Formula: True Positives / (True Positives + False Positives)
        Answers: "When we predict something is positive, how often are we right?"
        Recall: Out of all the actual positive instances (1), how many did we correctly identify
        Formula: True Positives / (True Positives + False Negatives)
        Answers: "Out of all actual positives, how many did we catch?"
        F1-score: The harmonic mean of precision and recall
        Formula: 2 * (Precision * Recall) / (Precision + Recall)
        Provides a single score that balances both metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average=average, zero_division=0
    )
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "support_right": sum(y_true),
        "support_false": len(y_true) - sum(y_true),
    }

    return metrics


def log_scores_distributions(y_true, y_pred, lof_scores):
    """
    Generate LOF's score distributions, separating correct and incorrect predictions.
    Handles cases where there might be no incorrect predictions.

    Args:
        y_true: True labels (array of 0/1 or False/True)
        y_pred: Predicted labels (array of 0/1 or False/True)
        lof_scores: LOF scores for each prediction
    """
    # Convert to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    lof_scores = np.array(lof_scores)

    # Get masks for different conditions
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    # Separate scores based on true labels and correctness
    true_positive_scores = filter_scores(lof_scores[(y_true == 1) & correct_mask])
    true_negative_scores = filter_scores(lof_scores[(y_true == 0) & correct_mask])
    false_positive_scores = filter_scores(lof_scores[(y_true == 0) & incorrect_mask])
    false_negative_scores = filter_scores(lof_scores[(y_true == 1) & incorrect_mask])

    # Create figure with correct predictions
    figure_correct = create_histogram_figure(
        true_positive_scores,
        true_negative_scores,
        "Correct Predictions LOF Scores",
        "True Positives",
        "True Negatives",
    )

    figure_incorrect = create_histogram_figure(
        false_negative_scores,
        false_positive_scores,
        "Incorrect Predictions LOF Scores",
        "False Negatives",
        "False Positives",
    )

    return figure_correct, figure_incorrect


def filter_scores(data):
    """
    Filter outliers from the data using IQR method.
    Handles empty arrays and arrays with insufficient data for quartile calculation.
    """
    if len(data) == 0:
        return np.array([])

    if len(data) < 4:  # Not enough data for meaningful quartile calculation
        return np.array(data)

    data = np.array(data)
    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Define the bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data


def create_histogram_figure(scores1, scores2, title, label1, label2):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate optimal bins
    all_scores = np.concatenate([scores1, scores2])
    bins = np.linspace(min(all_scores), max(all_scores), 30)

    # Create histograms with enhanced styling
    ax.hist(
        scores1,
        bins=bins,
        alpha=0.6,
        color="forestgreen",
        label=label1,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        scores2,
        bins=bins,
        alpha=0.6,
        color="lightcoral",
        label=label2,
        edgecolor="black",
        linewidth=0.5,
    )

    # Enhance the plot
    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    ax.set_xlabel("LOF Score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Adjust layout
    plt.tight_layout()
    return fig


def validate_folder_name(folder_name):
    """Validate that folder name matches the expected format."""
    pattern = r"test_seed_\d+"
    return bool(re.match(pattern, os.path.basename(folder_name)))


def find_tensorboard_file(folder_path):
    """Find the tensorboard events file in the given folder."""
    pattern = "events.out.tfevents.*"
    matches = glob.glob(os.path.join(folder_path, pattern))
    if not matches:
        raise FileNotFoundError(f"No tensorboard file found in {folder_path}")
    return matches[0]


def extract_metrics_from_tensorboard(tb_file):
    """Extract specific metrics from tensorboard file."""
    ea = event_accumulator.EventAccumulator(tb_file)
    ea.Reload()

    metrics = {}
    for metric in ["test_f1", "test_recall", "test_precision", "test_accuracy"]:
        try:
            # Get the last value for each metric
            values = ea.Scalars(metric)
            metrics[metric] = values[-1].value
        except KeyError:
            warnings.warn(f"Metric {metric} not found in {tb_file}")
            metrics[metric] = None

    return metrics


def process_all_seeds(root_path):
    """Process all seed folders and collect metrics."""
    all_metrics = []

    # Get all subdirectories
    subdirs = [d for d in glob.glob(os.path.join(root_path, "*")) if os.path.isdir(d)]

    for subdir in subdirs:
        if not validate_folder_name(subdir):
            warnings.warn(f"Skipping invalid folder name: {subdir}")
            continue

        try:
            tb_file = find_tensorboard_file(subdir)
            metrics = extract_metrics_from_tensorboard(tb_file)
            all_metrics.append(metrics)
        except Exception as e:
            warnings.warn(f"Error processing {subdir}: {str(e)}")

    return all_metrics


def calculate_statistics(metrics_list):
    """Calculate average and standard deviation for each metric."""
    if not metrics_list:
        raise ValueError("No valid metrics found")

    # Convert list of dictionaries to dictionary of lists
    keys = metrics_list[0].keys()
    metrics_by_name = {
        metric: [d[metric] for d in metrics_list if d[metric] is not None]
        for metric in keys
        if "support" not in metric
    }

    # Calculate statistics
    stats = {}
    for metric, values in metrics_by_name.items():
        if values:
            avg = np.mean(values)
            std = np.std(values)
            stats[metric] = f"{avg:.3f}+-{std:.3f}"
        else:
            stats[metric] = "N/A"

    return stats
