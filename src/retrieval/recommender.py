from __future__ import annotations

from typing import Any

import numpy as np

from src.inference import MovieInference
from src.retrieval.faiss_index import MovieIndex


class MovieRecommender:
    """
    High-level movie recommendation service.
    """

    def __init__(
        self,
        inference: MovieInference,
        index: MovieIndex,
    ) -> None:

        self.inference = inference
        self.index = index

    def recommend(
        self,
        movie_text: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Recommend similar movies.

        Args:
            movie_text:
                Complete movie description.

            top_k:
                Number of recommendations.

        Returns:
            List of recommended movies.
        """

        embedding = self.inference.encode(
            movie_text
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        return self.index.search(
            embedding=embedding,
            top_k=top_k,
        )