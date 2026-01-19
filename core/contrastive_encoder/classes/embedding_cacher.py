import torch
from tqdm import tqdm
import hashlib
from typing import Optional, Dict
import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the embedding cache.

        Args:
            cache_dir (str, optional): Directory to persist cache to disk.
                If None, cache is kept only in memory.
        """
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_dir = cache_dir

    def compute_hash(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> str:
        """
        Compute a unique hash for the input tensors.
        """
        # Concatenate the tensors and convert to numpy for consistent hashing
        combined = np.concatenate(
            [input_ids.cpu().numpy().flatten(), attention_mask.cpu().numpy().flatten()]
        )

        # Create hash
        return hashlib.md5(combined.tobytes()).hexdigest()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve embedding from cache if it exists."""
        return self.cache.get(key)

    def put(self, key: str, embedding: torch.Tensor):
        """Store embedding in cache."""
        self.cache[key] = embedding
