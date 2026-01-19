import os
from typing import Dict, List
import torch
import numpy as np
from tqdm import tqdm
from numpy.random import default_rng
from core.common.utils import read_json
from collections import defaultdict
from typing import Optional, Tuple


class OptimizedBatchProcessor:
    def __init__(self, unique_rules_sample: List[str]):
        """
        Initialize the batch processor with a list of unique rules.
        Args:
            unique_rules_sample: List of unique rules to process
        """
        self.unique_rules = np.array(
            unique_rules_sample
        )  # Convert to numpy array for faster indexing
        # Create bidirectional mappings using numpy arrays for faster lookup
        self.index_2_rule = {idx: rule for idx, rule in enumerate(unique_rules_sample)}
        self.rule_2_index = {rule: idx for idx, rule in self.index_2_rule.items()}
        self._embedding_cache = None
        self.sampling_matrix = None

    @property
    def embedding_cache(self) -> Optional[Dict]:
        """Property to access embedding cache with error checking."""
        if self._embedding_cache is None:
            raise ValueError(
                "Embeddings not precomputed. Call preprocess_embeddings first."
            )
        return self._embedding_cache

    def preprocess_embeddings(self, embeddings: Dict, is_token: bool) -> None:
        """
        Preprocess and cache embeddings as torch tensors.

        Args:
            embeddings: Dictionary containing sentence and rule embeddings
            is_token: Boolean indicating if embeddings are tokenized
        """
        self._embedding_cache = defaultdict(dict)
        for key in ["sentences", "rules"]:
            for k, v in embeddings[key].items():
                if not is_token:
                    # Optimize tensor creation by pre-allocating correct shape
                    tensor = (
                        torch.tensor(v, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
                    )
                else:
                    tensor = v
                self._embedding_cache[key][str(k)] = tensor

    def _create_invalid_indices_map(
        self, mr_2_ps: Dict[str, str], ps_2_mrs: Dict[str, set], n_samples: int
    ) -> Dict[int, set]:
        """
        Pre-compute invalid indices for negative sampling.

        Args:
            mr_2_ps: Mapping from macro rules to positive samples
            ps_2_mrs: Mapping from positive samples to macro rules
            n_samples: Number of samples to process

        Returns:
            Dictionary mapping sample indices to their invalid indices
        """
        invalid_indices = {}
        for i, rule in tqdm(
            self.index_2_rule.items(),
            desc="Processing samples to identify positive indices...",
        ):
            # Get all rules that share positive samples with current rule
            current_ps = mr_2_ps[rule]
            positive_rules = ps_2_mrs[current_ps]
            invalid_indices[i] = {self.rule_2_index[mr] for mr in positive_rules}

        return invalid_indices

    def perform_negative_sampling(
        self,
        n_samples: int,
        indices_set: set,
        invalid_indices: Dict[int, set],
        expanded_n_negative_samples: int,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Optimized negative sampling implementation using vectorized operations.

        Args:
            n_samples: Number of samples to generate
            indices_set: Set of all possible indices
            invalid_indices: Dictionary mapping indices to their invalid indices
            expanded_n_negative_samples: Number of negative samples needed per positive
            seed: Random seed for reproducibility

        Returns:
            Matrix containing negative samples
        """
        rng = default_rng(seed)
        # Pre-allocate matrix with -1 as padding value
        matrix = np.full((n_samples, expanded_n_negative_samples), -1, dtype=np.int64)
        all_indices = np.array(list(indices_set))
        # Vectorized mask operations
        mask = np.ones(len(all_indices), dtype=bool)
        for i in tqdm(range(n_samples), desc="Sampling candidate negative samples..."):
            mask[:] = True  # Reset mask efficiently
            mask[list(invalid_indices[i])] = False
            # Get valid indices using boolean indexing
            valid_indices = all_indices[mask]
            if len(valid_indices) > 0:
                n_samples_to_take = min(len(valid_indices), expanded_n_negative_samples)
                selected_indices = rng.choice(
                    valid_indices, size=n_samples_to_take, replace=False
                )
                matrix[i, :n_samples_to_take] = selected_indices
        return matrix

    def create_negative_sampling_matrix(
        self,
        mr_2_ps: Dict[str, str],
        ps_2_mrs: Dict[str, set],
        n_negative_samplings: int,
        seed: int,
        cache_dir: Optional[str],
        logger: object,
        oversample_factor: int = 4,
    ) -> None:
        """
        Create or load cached negative sampling matrix.

        Args:
            mr_2_ps: Mapping from macro rules to positive samples
            ps_2_mrs: Mapping from positive samples to macro rules
            n_negative_samplings: Number of negative samples per positive
            seed: Random seed for reproducibility
            cache_dir: Directory for caching results
            logger: Logger object
            oversample_factor: Factor for oversampling negatives
        """
        cache_file = (
            os.path.join(cache_dir, "matrix_negative_samplings.npz")
            if cache_dir
            else None
        )
        if cache_file and os.path.isfile(cache_file):
            logger.debug(
                "\tLoading the negative sample matrix from cache at %s", cache_file
            )
            data = np.load(cache_file, allow_pickle=True)
            self.sampling_matrix = data["sampling_matrix"]
            self.index_2_rule = {
                int(k): v for k, v in data["index_2_rule"].item().items()
            }
            self.rule_2_index = data["rule_2_index"].item()
        else:
            logger.debug("\tComputing the negative sample matrix from scratch...")
            n_samples = len(self.unique_rules)
            expanded_n_negative_samples = n_negative_samplings * oversample_factor
            # Pre-compute invalid indices
            invalid_indices = self._create_invalid_indices_map(
                mr_2_ps, ps_2_mrs, n_samples
            )
            indices_set = set(range(n_samples))
            self.sampling_matrix = self.perform_negative_sampling(
                n_samples=n_samples,
                indices_set=indices_set,
                invalid_indices=invalid_indices,
                expanded_n_negative_samples=expanded_n_negative_samples,
                seed=seed,
            )
            if cache_file:
                logger.debug("\tCaching negative sample matrix at %s", cache_file)
                np.savez(
                    cache_file,
                    sampling_matrix=self.sampling_matrix,
                    index_2_rule={str(k): v for k, v in self.index_2_rule.items()},
                    rule_2_index=self.rule_2_index,
                )

    def _process_batch_embeddings(
        self,
        batch_size: int,
        sampled_rules: np.ndarray,
        mr_to_hs: Dict[str, str],
        mr_2_rules: Dict[str, set],
        embed_dim: int,
        n_negative_samplings: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process batch embeddings efficiently.

        Args:
            batch_size: Size of the batch
            sampled_rules: Array of sampled rules for the batch
            mr_to_hs: Mapping from macro rules to hash strings
            mr_2_rules: Mapping from macro rules to rule sets
            embed_dim: Embedding dimension
            n_negative_samplings: Number of negative samples
        Returns:
            Tuple of processed tensors
        """
        # Pre-allocate tensors
        input_embeddings = torch.zeros(
            (batch_size, 1 + n_negative_samplings, 2, embed_dim)
        )
        padding_tensors = torch.zeros((batch_size, 1 + n_negative_samplings))
        # Process rules
        max_rules = max(len(mr_2_rules[rule]) for rule in sampled_rules)
        rule_embeddings = torch.zeros((batch_size, max_rules, 2, embed_dim))
        padding_mask_rule = torch.zeros(batch_size, max_rules, dtype=torch.bool)
        # Process positive embeddings
        for batch_idx, pos_hash in enumerate(sampled_rules):
            input_embeddings[batch_idx, 0] = self.embedding_cache["sentences"][
                str(mr_to_hs[pos_hash])
            ]
        return input_embeddings, padding_tensors, rule_embeddings, padding_mask_rule

    def batch_extract_training_samples(
        self,
        batch_indices: List[int],
        mr_to_hs: Dict[str, str],
        mr_2_rules: Dict[str, set],
        mr_2_ps: Dict[str, str],
        pos_hash_2_other_positives: Dict[str, str],
        n_negative_samplings: int,
        partition: str = "train",
        base_seed: int = 42,
    ) -> Dict[str, torch.Tensor]:
        """
        Optimized version of batch_extract_training_samples with improved vectorization.
        """
        # Create batch-specific RNG
        batch_seed = base_seed + hash(tuple(batch_indices)) % 10000
        rng = np.random.default_rng(batch_seed)
        # Get batch size and embedding dimension
        batch_size = len(batch_indices)
        embed_dim = next(iter(self.embedding_cache["sentences"].values())).shape[-1]
        # Get the extracted batch samples (remember: each sample is an identifier for a tuple rule+example)
        batch_samples = self.unique_rules[batch_indices]
        # Initialize tensors for embeddings
        input_embeddings, padding_tensors, rule_embeddings, padding_mask_rule = (
            self._process_batch_embeddings(
                batch_size,
                batch_samples,
                mr_to_hs,
                mr_2_rules,
                embed_dim,
                n_negative_samplings,
            )
        )
        #### Now, preparing what we need to compute the labels later on (the positive hashes and the sampled ones)
        # Retrieve the positive hashes per batch
        positive_sample_hashes = np.array(
            [mr_2_ps[rule] for rule in batch_samples], dtype=object
        )
        # Initialize sampled rules array
        sampled_rules = np.zeros((batch_size, 1 + n_negative_samplings), dtype=object)
        # As we know that each rule is assigned a positive samples by construction, fill the first column
        sampled_rules[:, 0] = positive_sample_hashes
        #### Process negative samples
        # Get available indices for negative sampling
        available_indices = self.sampling_matrix[batch_indices]
        valid_masks = available_indices != -1  # remove padding values
        # Process negative samples for each batch
        for batch_idx, (valid_mask, avail_indices) in enumerate(
            zip(valid_masks, available_indices)
        ):
            valid_indices = avail_indices[valid_mask]
            # Select negative samples based on partition
            # During training, we extract random negative samples from the available (so that we do not over-fit)
            if partition == "train":
                if len(valid_indices) > n_negative_samplings:
                    selected_indices = rng.choice(
                        valid_indices, size=n_negative_samplings, replace=False
                    )
                else:
                    selected_indices = valid_indices[:n_negative_samplings]
            else:  # validation or control
                selected_indices = valid_indices[:n_negative_samplings]
            # Update padding tensor
            padding_tensors[batch_idx, : 1 + len(selected_indices)] = 1
            # Process negative samples
            negative_macro_rules = self.unique_rules[selected_indices]
            # Update embeddings and rules
            input_embeddings[batch_idx, 1 : 1 + len(selected_indices)] = torch.stack(
                [
                    self.embedding_cache["sentences"][str(mr_to_hs[neg_hash])]
                    for neg_hash in negative_macro_rules
                ]
            )
            sampled_rules[batch_idx, 1 : 1 + len(selected_indices)] = [
                mr_2_ps[mr] for mr in negative_macro_rules
            ]

        # Vectorized rule embedding processing
        for batch_idx, rule in enumerate(batch_samples):
            rule_hashes = mr_2_rules[rule]
            rule_embeddings[batch_idx, : len(rule_hashes)] = torch.stack(
                [
                    self.embedding_cache["rules"][str(rule_hash)]
                    for rule_hash in rule_hashes
                ]
            )
            padding_mask_rule[batch_idx, : len(rule_hashes)] = True

        # Construct label matrix
        if partition == "control":
            # For control, only first sample is positive
            rule_vs_samples_labels = torch.zeros(batch_size, 1 + n_negative_samplings)
            rule_vs_rules_labels = torch.zeros(batch_size, batch_size)
            rule_vs_samples_labels[:, 0] = 1
        else:
            # For training and validation, we compute in-batch loss
            # Hence, we must consider that the negative samples for a given positive might be positives for other samples
            flatten_sampled_rules = sampled_rules.reshape(-1)
            rule_vs_samples_labels = torch.zeros(
                positive_sample_hashes.shape[0], len(flatten_sampled_rules)
            )
            # Create a matrix of positive sets for each batch sample
            positive_sets = [
                pos_hash_2_other_positives[ph] for ph in positive_sample_hashes
            ]
            # Vectorized comparison using numpy operations
            #    Remember: we are checking if the sampled negative sentences are positives for some rule
            positive_matrix = np.array(
                [
                    [sampled_hash in pos_set for sampled_hash in flatten_sampled_rules]
                    for pos_set in positive_sets
                ]
            )
            # Convert to tensor
            rule_vs_samples_labels = torch.from_numpy(positive_matrix).float()
            # Now, compare the rules against each other
            rule_vs_rules_labels = torch.tensor(
                [
                    [pos_hash in pos_set for pos_hash in positive_sample_hashes]
                    for pos_set in positive_sets
                ]
            ).float()
            # Fill the diagonal with zeros
            # diagonal_indices = torch.arange(rule_vs_rules_labels.size(0))
            # rule_vs_rules_labels[diagonal_indices, diagonal_indices] = 0

        return {
            "input_embedding": input_embeddings,
            "padding_input_embedding": padding_tensors,
            "rule_embeddings": rule_embeddings,
            "labels": rule_vs_samples_labels,
            "rules_vs_rules_labels": rule_vs_rules_labels,
            "padding_mask_rule": padding_mask_rule,
        }


def get_embeddings(paths_embeddings):
    paths_embeddings = (
        [paths_embeddings] if isinstance(paths_embeddings, str) else paths_embeddings
    )
    embeddings = {}
    for path_embeddings in paths_embeddings:
        for file in os.listdir(path_embeddings):
            if os.path.splitext(file)[1] == ".json":
                file_key = os.path.splitext(file)[0]
                file_path = os.path.join(path_embeddings, file)
                file_data = read_json(file_path)
                # Update existing dictionary or add a new one
                if file_key in embeddings:
                    embeddings[file_key].update(file_data)
                else:
                    embeddings[file_key] = file_data
    return embeddings
