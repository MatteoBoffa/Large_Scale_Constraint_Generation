from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def compute_metrics(y_true, y_pred, invalid_label="invalid"):
    """
    Compute binary classification metrics under two evaluation regimes
    in the presence of invalid (abstaining) predictions.

    This function reports:
    1) Worst-case metrics: invalid predictions are penalized as maximally
       incorrect by assigning them the opposite of the true label. This
       preserves full dataset support and enables fair comparison across
       models with different invalid rates.
    2) Best-case metrics: metrics are computed only on valid predictions
       (i.e., invalid predictions are excluded), reflecting conditional
       performance when the model produces a usable output.

    The returned dictionary contains both sets of metrics, distinguished
    by the suffixes '_worst_case' and '_best_case'.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.

    y_pred : array-like of shape (n_samples,)
        Model predictions. May contain binary labels as well as an
        `invalid_label` indicating abstention or invalid output.

    invalid_label : hashable, default="invalid"
        Value in `y_pred` indicating an invalid or abstained prediction.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, and F1 score
        (in percentage) for both worst-case and best-case evaluations.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=object)
    worst_case_y_pred = punish_invalid(
        y_true=y_true, y_pred=y_pred, invalid_label=invalid_label
    )
    worst_case_metrics = compute_score(
        y_true=y_true, y_pred=worst_case_y_pred, case="_worst_case"
    )
    mask = y_pred != invalid_label
    # Case in which we only have invalid values
    if not np.any(mask):
        best_case_metrics = {
            "accuracy_best_case": 0.0,
            "precision_best_case": 0.0,
            "recall_best_case": 0.0,
            "f1_score_best_case": 0.0,
        }
    else:
        y_true_best = y_true[mask].astype(bool)
        y_pred_best = np.asarray(y_pred[mask], dtype=bool)
        best_case_metrics = compute_score(
            y_true=y_true_best, y_pred=y_pred_best, case="_best_case"
        )

    return worst_case_metrics | best_case_metrics


def punish_invalid(y_true, y_pred, invalid_label):
    """
    Replace invalid predictions with the opposite of the true label.

    This transformation penalizes invalid predictions as maximally wrong
    on a per-sample basis, ensuring that all samples contribute to the
    evaluation metrics and that abstentions cannot artificially inflate
    performance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.

    y_pred : array-like of shape (n_samples,)
        Model predictions, possibly containing invalid labels.

    invalid_label : hashable
        Value indicating an invalid or abstained prediction.

    Returns
    -------
    ndarray of bool
        Binary predictions with invalid values replaced by the opposite
        of the corresponding true labels.
    """
    punished = y_pred.copy()
    mask = punished == invalid_label
    # Opposite wrt the correct label
    punished[mask] = np.logical_not(y_true[mask])
    return punished.astype(bool)


def compute_score(y_true, y_pred, case=""):
    """
    Compute standard binary classification metrics.

    Metrics include accuracy, precision, recall, and F1 score, returned
    as percentages. An optional suffix can be appended to metric names
    to distinguish different evaluation cases.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.

    y_pred : array-like of shape (n_samples,)
        Binary predictions.

    case : str, default=""
        Suffix appended to metric names (e.g., '_worst_case').

    Returns
    -------
    dict
        Dictionary of computed metrics in percentage.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {
        f"accuracy{case}": accuracy * 100,
        f"precision{case}": precision * 100,
        f"recall{case}": recall * 100,
        f"f1_score{case}": f1 * 100,
    }
