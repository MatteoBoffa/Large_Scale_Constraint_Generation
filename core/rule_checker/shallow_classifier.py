import random
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, TypeVar, Type, Optional

import numpy as np
import torch
from tqdm import tqdm

from core.common.data_handler import pad_ragged_tensor_lists
from sklearn.metrics import precision_recall_curve

T = TypeVar("T", bound="Shallow_Classifier")


class Shallow_Classifier:
    """Shared logic for rule compliance checkers based on rule/sentence embeddings.

    Subclasses must implement `tune_hyperparameters`.
    """

    def __init__(self, rule_embedder):
        """Initialize the checker.

        Args:
            rule_embedder: Model that embeds rules and sentences. Must expose:
                - encode_rules(rule_tensor, padding_mask) -> Tensor [B, D]
                - encode_sentences(sentence_tensor) -> Tensor [B, D] or [B, 1, D]
        """
        self.rule_embedder = rule_embedder
        self.best_params_ = None
        self.classifier = None
        self.confidence_threshold_ = None

    # --- Get confidence threshold -------------------------------------------------
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold_

    # --- Embedding helpers -------------------------------------------------
    def _embed_rule(self, rule: torch.Tensor, padding_mask: torch.Tensor) -> np.ndarray:
        """Embed a rule and convert to numpy array."""
        with torch.no_grad():
            return self.rule_embedder.encode_rules(rule, padding_mask).cpu().numpy()

    def _embed_sentence(self, sentence: torch.Tensor) -> np.ndarray:
        """Embed a sentence and convert to numpy array."""
        with torch.no_grad():
            return self.rule_embedder.encode_sentences(sentence).cpu().numpy()

    def _prepare_embeddings(
        self,
        sentences: torch.tensor,
        rules: torch.tensor,
        padding_mask: torch.tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare rule embeddings and related data structures."""
        # We need to add the dimension N to make the encoder work
        sentences = sentences.unsqueeze(1)  # [Bx2xH_in] > [Bx1x2xH_in]
        # Remember: remove the N samples dimension (used only for training)
        sent_emb = self._embed_sentence(sentences).squeeze(1)
        rule_embeddings = (
            self.rule_embedder.encode_rules(
                rule_embeddings=rules, padding_mask_rule=padding_mask
            )
            .detach()
            .cpu()
            .numpy()
        )
        return rule_embeddings, sent_emb

    # --- Training data preparation ----------------------------------------
    def _prepare_training_data(
        self,
        rules: Dict[str, List[torch.Tensor]],
        positive_samples: Dict[str, List[torch.Tensor]],
        device: str,
        percentage_negatives: float,
        batch_size: int,
        seed: int,
    ):
        """Prepare training data for the model."""
        # Set seeds
        random.seed(seed)
        np.random.seed(seed=seed)
        rules_input, sentences_input, labels = [], [], []

        for rule_id, rule in tqdm(rules.items(), desc="Preparing training data"):
            # Process positive samples for this rule
            pos_samples = positive_samples[rule_id]
            # Sample negative examples from other rules
            other_rules = [r for r in positive_samples.keys() if r != rule_id]
            # Define the number of negative samples to ensure good coverage
            #   Ideally, we would like to use all the other samples as negatives
            #   However, this is not feasible for large datasets: we will hence sample 1 sample for a fixed number of rules
            #   Such fixed number will be at most a percentage of the positive rules
            n_negatives_rules = min(
                int(percentage_negatives * len(pos_samples)), len(other_rules)
            )
            # Now, extract 1 negatives sample for the identified negatives rules
            random_indices = np.random.choice(
                len(other_rules), size=n_negatives_rules, replace=False
            )
            # Select the negative examples we will sample from
            selected_negative_rules = [other_rules[i] for i in random_indices]
            # Sample negative samples from other rules
            negative_sentences = []
            for negative_rule_id in selected_negative_rules:
                other_samples = positive_samples[negative_rule_id]
                # Randomly select 1 element
                negative_sentences.append(random.choice(other_samples))
            # Now append the positive elements
            for sent in pos_samples:
                rules_input.append(rule)
                sentences_input.append(sent)
                labels.append(1)
            # And the negatives
            for sent in negative_sentences:
                rules_input.append(rule)
                sentences_input.append(sent)
                labels.append(0)

        # Embed rules and sentences
        X_train = []
        for i in tqdm(range(0, len(rules_input), batch_size), desc="Embedding data"):
            batch_slice = slice(i, i + batch_size)
            rules_batch = rules_input[batch_slice]

            # Handle variable-length rules
            max_rules = max(len(rule) for rule in rules_batch)
            embedding_dim = rules_batch[0][0].shape[-1]

            rule_embeddings = torch.zeros(
                (len(rules_batch), max_rules, 2, embedding_dim), device=device
            )
            padding_mask = torch.zeros(
                len(rules_batch), max_rules, dtype=torch.bool, device=device
            )

            # Fill in the actual embeddings
            for j, rule in enumerate(rules_batch):
                rule_embeddings[j, : len(rule)] = torch.stack(rule)
                padding_mask[j, : len(rule)] = True

            # Get embeddings
            rules_emb = self._embed_rule(rule_embeddings, padding_mask)
            sentences_batch = (
                torch.stack(sentences_input[batch_slice]).to(device).unsqueeze(1)
            )
            sentences_emb = self._embed_sentence(sentences_batch).squeeze(1)

            # Combine features
            batch_features = np.concatenate([rules_emb, sentences_emb], axis=1)
            X_train.append(batch_features)

        X_train = np.vstack(X_train)
        y_train = np.array(labels)

        return X_train, y_train

    # --- Model lifecycle ---------------------------------------------------

    def tune_hyperparameters(self, X_train, y_train, cv=3, seed=42, n_proc=1):
        """Subclasses must implement."""
        raise NotImplementedError

    def fit(
        self,
        rules: Dict[str, List[torch.Tensor]],
        positive_samples: Dict[str, List[torch.Tensor]],
        device: str = "cpu",
        percentage_negatives: float = 0.5,
        batch_size: int = 32,
        cv: int = 3,
        n_proc: int = 1,
        seed: int = 42,
        logger: object = None,
    ) -> None:
        """Fit the checker classifier.

        Args:
            rules: mapping rule_id -> list of rule tensors (ragged)
            positive_samples: mapping rule_id -> list of positive sentence tensors
            device: torch device
            percentage_negatives: negatives per positive
            batch_size: embedding batch size
            cv: cross-validation folds
            n_proc: parallel CV jobs
            seed: random seed (data sampling + CV shuffling where supported)
            logger: optional logger with .info
        """
        logger.info("\tPreparing training data...")
        X_train, y_train = self._prepare_training_data(
            rules, positive_samples, device, percentage_negatives, batch_size, seed
        )

        # Tune hyperparameters
        logger.info("Tuning hyperparameters...")
        best_params, _ = self.tune_hyperparameters(
            X_train, y_train, cv=cv, seed=seed, n_proc=n_proc
        )
        logger.info("\nBest parameters found:")
        logger.info(best_params)

    def find_optimal_confidence_score(self, y_true, p, min_precision=0.8):
        """
        Select an optimal classification threshold that maximizes recall
        while enforcing a minimum precision constraint.

        The function operates on predicted probabilities (typically obtained
        from an out-of-fold validation set or a held-out validation set) and
        searches over all possible decision thresholds derived from the
        precision–recall curve.

        Among all thresholds achieving a precision greater than or equal to
        `min_precision`, it selects the one with the highest recall.
        If no threshold satisfies the minimum precision requirement, the
        threshold yielding the highest achievable precision is selected
        instead.

        Args:
            y_true (array-like of shape (n_samples,)):
                Ground-truth binary labels (0 or 1) for each sample.

            p (array-like of shape (n_samples,)):
                Predicted probabilities for the positive class (class label 1),
                e.g. the second column of `predict_proba`.

            min_precision (float, optional):
                Minimum acceptable precision (positive predictive value).
                Must be in the interval (0, 1]. Defaults to 0.8.

        Returns:
            tuple:
                A tuple containing:
                - best_threshold (float): The selected probability threshold.
                - best_precision (float): Precision achieved at this threshold.
                - best_recall (float): Recall achieved at this threshold.

        Notes:
            - This function should be used with out-of-sample probabilities
            (e.g. out-of-fold predictions) to avoid optimistic bias.
            - The selected threshold represents an operating point optimized
            for high recall under a precision constraint, not a globally
            optimal classifier.
            - For models with poorly calibrated probabilities (e.g. random
            forests), probability calibration may improve threshold stability.
        """

        precision, recall, thresholds = precision_recall_curve(y_true, p)

        # precision and recall have length len(thresholds)+1
        # thresholds correspond to precision[1:], recall[1:]
        precision_t = precision[1:]
        recall_t = recall[1:]

        valid = precision_t >= min_precision
        if not np.any(valid):
            # can't meet the requirement; pick the best precision you can, or raise model quality
            best_idx = np.argmax(precision_t)
        else:
            # among valid thresholds, choose the one with highest recall
            best_idx = np.argmax(np.where(valid, recall_t, -np.inf))

        best_threshold = thresholds[best_idx]
        best_precision = precision_t[best_idx]
        best_recall = recall_t[best_idx]

        return best_threshold, best_precision, best_recall

    def check_compliance(
        self,
        sentences: List[torch.tensor],
        rules: list,
        confidence_score: float = 0.5,
        device: str = "cpu",
    ) -> tuple:
        """Check compliance of batches of sentences with rules."""

        # Remember: convert list of tensors to tensor
        # Notice: each rule contains a variable number of concepts > we need padding
        # Created tensors will have shape padded_rules = [B, Lmax, 2, H_in] and padding_mask = [B, Lmax]
        padded_rules, padding_mask = pad_ragged_tensor_lists(rules, device=device)
        # Also create a tensor for sentences
        sentences = torch.stack(sentences).to(device)
        # We will obtain embeddings of shape [B, H_emb] for both rules and sentences
        rule_embeddings, sent_emb = self._prepare_embeddings(
            sentences,
            padded_rules,
            padding_mask,
        )
        # Concatenate rule and sentence embeddings so that classifier can process them
        X = np.concatenate([rule_embeddings, sent_emb], axis=1)
        scores = self.classifier.predict_proba(X)[:, 1]
        is_compliant = (scores >= confidence_score).astype(bool).tolist()
        return is_compliant, scores

    def save(self, path: str) -> None:
        """Save the fitted checker to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {"best_params": self.best_params_}

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        with open(path / "classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)

    def store_confidence_threshold(self, best_threshold: float, path: str) -> None:
        """
        Store the selected confidence threshold to disk.

        Args:
            best_threshold (float):
                Probability threshold used to convert predicted probabilities
                into binary predictions.

            path (str):
                Directory where the threshold will be stored.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        payload = {"confidence_threshold": float(best_threshold)}

        with open(path / "confidence_threshold.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls: Type[T], path: str, rule_embedder, device: Optional[str] = None) -> T:
        """Load a fitted checker from disk."""
        path = Path(path)

        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load it on GPU once
        if device is not None:
            rule_embedder = rule_embedder.to(device)
        # Put in eval mode
        rule_embedder.eval()

        instance = cls(rule_embedder=rule_embedder)
        instance.best_params_ = metadata["best_params"]

        with open(path / "classifier.pkl", "rb") as f:
            instance.classifier = pickle.load(f)

        threshold_path = path / "confidence_threshold.json"
        if threshold_path.exists():
            with open(threshold_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            instance.confidence_threshold_ = payload["confidence_threshold"]
        else:
            instance.confidence_threshold_ = 0.5

        return instance
