from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from config.settings import (
    MODELS_DIR,
    TOP_K,
)


class MovieIndex:

    def __init__(self):

        self.index = None
        self.movies = None

    def build(
        self,
        embeddings: np.ndarray,
    ) -> None:
        """
        Build FAISS index.
        """

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        faiss.normalize_L2(
            embeddings
        )

        self.index.add(
            embeddings
        )

    def save(
        self,
        path: Path,
    ) -> None:

        faiss.write_index(
            self.index,
            str(path),
        )

    def load(
        self,
        path: Path,
    ) -> None:

        self.index = faiss.read_index(
            str(path),
        )

    def load_movies(
        self,
        dataframe: pd.DataFrame,
    ):

        self.movies = dataframe.reset_index(
            drop=True
        )

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = TOP_K,
    ):

        embedding = embedding.reshape(
            1,
            -1,
        )

        faiss.normalize_L2(
            embedding
        )

        scores, indices = self.index.search(
            embedding,
            top_k,
        )

        recommendations = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

            movie = self.movies.iloc[idx]

            recommendations.append(
                {

                    "title": movie["title"],

                    "score": float(score),

                    "genres": movie["genres_text"],

                    "year": movie["release_year"],

                    "rating": movie["vote_average"],

                }
            )

        return recommendations