from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Retrieved:
    text: str
    score: float


class DenseForbiddenRAG:
    """
    For each sentence:
      - embed the sentence
      - embed the candidate pool (forbidden list for that sentence)
      - score and return top-k
    """

    def __init__(
        self,
        embed_model: str = "BAAI/bge-m3",
        reranker_model: Optional[
            str
        ] = None,  # e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"
        use_cosine: bool = True,
        dtype=np.float32,
        hf_cache=None,
        prompt_path=None,
    ):
        self.embedder = SentenceTransformer(embed_model, cache_folder=hf_cache)
        self.reranker = (
            CrossEncoder(reranker_model, cache_folder=hf_cache)
            if reranker_model
            else None
        )
        self.use_cosine = use_cosine
        self.dtype = dtype
        self.prompt = self.load_prompt(prompt_path) if prompt_path else None

    def load_prompt(self, prompt_path):
        with open(prompt_path) as f:
            prompt = json.load(f)
        return prompt["rules"]

    def topk_from_pool(
        self,
        sentence: str,
        forbidden_pool: List[str],
        top_k: int = 20,
        prefilter_k: int = 200,  # only used if reranker is enabled
        batch_size: int = 256,
    ) -> List[Retrieved]:
        if not forbidden_pool:
            return []

        # Embed query
        q = self.embedder.encode(
            [sentence],
            convert_to_numpy=True,
            normalize_embeddings=self.use_cosine,
        ).astype(self.dtype)[
            0
        ]  # (D,)

        prompt = self.prompt + " " if self.prompt else ""

        # Prepend prompt if any
        forbidden_pool = [prompt + el for el in forbidden_pool]

        # Embed pool (this is the expensive step per example)
        pool_emb = self.embedder.encode(
            forbidden_pool,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.use_cosine,
            show_progress_bar=False,
        ).astype(
            self.dtype
        )  # (n, D)

        # Score (cosine if normalized, otherwise dot-product similarity)
        scores = pool_emb @ q  # (n,)

        # Preselect top candidates
        k0 = min(prefilter_k if self.reranker else top_k, len(forbidden_pool))
        top_idx = np.argpartition(-scores, kth=k0 - 1)[:k0]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        if self.reranker:
            # Rerank only top k0
            cands = [forbidden_pool[i] for i in top_idx.tolist()]
            pairs = [(sentence, c) for c in cands]
            rr = self.reranker.predict(pairs)
            order = np.argsort(-rr)[: min(top_k, len(rr))]
            return [Retrieved(text=cands[i], score=float(rr[i])) for i in order]

        # No reranker
        top_idx = top_idx[: min(top_k, len(top_idx))]
        return [
            Retrieved(
                text=forbidden_pool[i].removeprefix(prompt),
                score=float(scores[i]),
            )
            for i in top_idx
        ]
