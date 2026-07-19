"""
Evaluation metrics for the Movie Recommendation Engine.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class RecommenderEvaluator:
    """
    Evaluate recommendation quality.
    """

    @staticmethod
    def precision_at_k(
        recommended: Sequence[int],
        relevant: Sequence[int],
        k: int,
    ) -> float:
        """
        Precision@K
        """

        recommended = recommended[:k]

        hits = len(
            set(recommended) &
            set(relevant)
        )

        return hits / k

    @staticmethod
    def recall_at_k(
        recommended: Sequence[int],
        relevant: Sequence[int],
        k: int,
    ) -> float:
        """
        Recall@K
        """

        if len(relevant) == 0:
            return 0.0

        recommended = recommended[:k]

        hits = len(
            set(recommended) &
            set(relevant)
        )

        return hits / len(relevant)

    @staticmethod
    def mean_reciprocal_rank(
        recommended: Sequence[int],
        relevant: Sequence[int],
    ) -> float:
        """
        Mean Reciprocal Rank (MRR)
        """

        for rank, movie in enumerate(
            recommended,
            start=1,
        ):

            if movie in relevant:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def ndcg_at_k(
        recommended: Sequence[int],
        relevant: Sequence[int],
        k: int,
    ) -> float:
        """
        NDCG@K
        """

        recommended = recommended[:k]

        dcg = 0.0

        for i, movie in enumerate(recommended):

            if movie in relevant:

                dcg += (
                    1.0 /
                    np.log2(i + 2)
                )

        ideal = min(
            len(relevant),
            k,
        )

        idcg = sum(

            1.0 / np.log2(i + 2)

            for i in range(ideal)

        )

        if idcg == 0:
            return 0.0

        return dcg / idcg