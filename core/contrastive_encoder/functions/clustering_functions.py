import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
from collections import defaultdict
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
from core.llm_encoder.encoder_func import mean_pooling


def validate_embeddings(embeddings_np):
    """
    Validate embeddings and handle zero vectors

    Args:
        embeddings_np: numpy array of embeddings

    Returns:
        tuple of (valid_embeddings, valid_indices)
        where valid_indices can be used to map back to original indices
    """
    # Calculate L2 norm for each vector
    norms = np.linalg.norm(embeddings_np, axis=1)

    # Find indices of non-zero vectors
    valid_indices = np.where(norms > 1e-8)[0]

    if len(valid_indices) == 0:
        raise ValueError("All vectors are zero vectors!")

    # Keep only non-zero vectors
    valid_embeddings = embeddings_np[valid_indices]

    return valid_embeddings, valid_indices


def get_pairs(groups):
    """
    Convert groups of items into pairs of items that belong to the same group

    Args:
        groups: List of lists, where each inner list contains items in the same group

    Returns:
        set of frozensets, where each frozenset contains a pair of items
    """
    pairs = set()
    for group in groups:
        for item1, item2 in combinations(group, 2):
            pairs.add(frozenset([item1, item2]))
    return pairs


def compute_sup_clustering_metrics(pred_groups, gt_groups):
    """
    Compute multiple clustering metrics by comparing predicted groups with ground truth

    Args:
        pred_groups: List of lists containing predicted groupings
        gt_groups: List of lists containing ground truth groupings

    Returns:
        dict containing various metrics
    """
    # Convert groups to pairs
    pred_pairs = get_pairs(pred_groups)
    gt_pairs = get_pairs(gt_groups)

    # Calculate metrics
    tp = len(pred_pairs & gt_pairs)  # True positives
    fp = len(pred_pairs - gt_pairs)  # False positives
    fn = len(gt_pairs - pred_pairs)  # False negatives

    # Compute metrics
    metrics = {}

    # Jaccard Index
    union = len(pred_pairs | gt_pairs)
    metrics["jaccard"] = round(tp / union, 3) if union > 0 else 0.0

    # F1 Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["f1"] = (
        round(2 * (precision * recall) / (precision + recall), 3)
        if (precision + recall) > 0
        else 0.0
    )

    # Rand Index
    n_items = len(set().union(*pred_groups))
    n_total_pairs = (n_items * (n_items - 1)) // 2
    tn = n_total_pairs - (tp + fp + fn)  # True negatives
    metrics["rand_index"] = (
        round((tp + tn) / n_total_pairs, 3) if n_total_pairs > 0 else 0.0
    )

    return metrics


def cluster_embeddings(rule_embeddings, input_concepts=None, gt_grouping=None):
    """
    Perform hierarchical clustering with parameter tuning and evaluation

    Args:
        rule_embeddings: tensor of shape (n_samples, embedding_dim)
        input_concepts: list of concept names. Default to None.
        gt_grouping (optional): ground truth grouping of concepts. Default to None.

    Returns:
        best_clusters: List of lists containing the best clustering result
        best_metrics: Dict containing the metrics for the best clustering
        best_params: Dict containing the parameters that gave the best result
    """
    # Convert embeddings to numpy if they're tensors
    if torch.is_tensor(rule_embeddings):
        embeddings_np = rule_embeddings.numpy()
    else:
        embeddings_np = rule_embeddings

    # Validate and handle zero vectors
    try:
        valid_embeddings, valid_indices = validate_embeddings(embeddings_np)
        valid_concepts = (
            [input_concepts[i] for i in valid_indices]
            if input_concepts is not None
            else []
        )
    except:
        # If all vectors are zero, return single cluster
        return [input_concepts], {"threshold": None}, 0.0, None

    best_score = -float("inf")
    best_labels = None
    best_params = None

    # Parameter grid for tuning
    distance_thresholds = np.linspace(0.1, 0.7, 30)

    # Try different parameters
    for threshold in distance_thresholds:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        )

        # Fit clustering on valid embeddings
        labels = clustering.fit_predict(valid_embeddings)

        # Only compute silhouette score if we have more than one cluster
        n_clusters = len(set(labels))
        if n_clusters > 1 and n_clusters < len(valid_embeddings):
            score = silhouette_score(valid_embeddings, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = {"threshold": round(float(threshold), 3)}

    # Convert labels to groups if we found a valid clustering
    if best_labels is not None:
        clusters = defaultdict(list)
        for concept, label in zip(valid_concepts, best_labels):
            clusters[label].append(concept)

        # Handle any zero vectors by putting them in their own cluster
        zero_vectors = set(input_concepts) - set(valid_concepts)
        if zero_vectors:
            clusters[max(clusters.keys()) + 1] = list(zero_vectors)

        clustered_groups = list(clusters.values())
    else:
        # If no valid clustering was found, put everything in one cluster
        clustered_groups = [input_concepts]
        best_score = 0.0
        best_params = {"threshold": None}

    if gt_grouping is not None:
        # Compute metrics
        sup_metrics = compute_sup_clustering_metrics(clustered_groups, gt_grouping)
    else:
        sup_metrics = None

    return clustered_groups, best_params, round(float(best_score), 3), sup_metrics


def process_synonym_sample(rule_embeddings, flat_concepts=None, gt_grouping=None):
    # Perform clustering and evaluation
    clusters, params, best_silh, sup_metrics = cluster_embeddings(
        rule_embeddings=rule_embeddings,
        input_concepts=flat_concepts,
        gt_grouping=gt_grouping,
    )
    return {
        "predicted_clusters": clusters,
        "best_silh": best_silh,
        "metrics": sup_metrics,
        "best_params": params,
    }


def extract_clusters(encoder, tokenizer, rules, device="cpu"):
    # Append prompt
    templated_rules = [f"The word in context is: {rule}" for rule in rules]
    # Tokenize rules
    tokenized_input = tokenizer(
        templated_rules, truncation=True, padding=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        model_output = encoder(**tokenized_input)
        embeddings = mean_pooling(model_output, tokenized_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
    clusters = process_synonym_sample(embeddings, flat_concepts=rules)[
        "predicted_clusters"
    ]
    return clusters


def reshape_embeddings_into_clusters(clusters, rule_emb, rules, device):
    MAX_CLUSTER_SIZE = max([len(cluster) for cluster in clusters])
    clustered_embeddings = torch.zeros(
        len(clusters),
        MAX_CLUSTER_SIZE,
        rule_emb.shape[-2],
        rule_emb.shape[-1],
        device=device,
    )
    concept2id = {concept: it for it, concept in enumerate(rules)}
    padding_mask = torch.zeros(len(clusters), MAX_CLUSTER_SIZE, device=device)
    for cluster_it, cluster in enumerate(clusters):
        for concept_it, concept in enumerate(cluster):
            corresponding_concept_id = concept2id[concept]
            corresponding_concept_emb = rule_emb[corresponding_concept_id, ...].squeeze(
                0
            )
            clustered_embeddings[cluster_it, concept_it] = corresponding_concept_emb
            padding_mask[cluster_it, concept_it] = 1
    return clustered_embeddings, padding_mask
