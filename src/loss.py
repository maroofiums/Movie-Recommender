from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss.

    Positive Pair (label = 1):
        Reduce distance.

    Negative Pair (label = 0):
        Increase distance beyond margin.
    """

    def __init__(
        self,
        margin: float = 1.0,
    ):

        super().__init__()

        self.margin = margin

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        distances = F.pairwise_distance(
            embedding_a,
            embedding_b,
        )

        positive_loss = labels * distances.pow(2)

        negative_loss = (
            (1 - labels)
            * torch.clamp(
                self.margin - distances,
                min=0.0,
            ).pow(2)
        )

        loss = positive_loss + negative_loss

        return loss.mean()


class CosineContrastiveLoss(nn.Module):
    """
    Contrastive loss using cosine similarity.

    Useful because movie embeddings are L2-normalized.
    """

    def __init__(
        self,
        margin: float = 0.5,
    ):

        super().__init__()

        self.margin = margin

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        similarity = F.cosine_similarity(
            embedding_a,
            embedding_b,
        )

        positive_loss = labels * (
            1 - similarity
        )

        negative_loss = (
            (1 - labels)
            * F.relu(
                similarity - self.margin
            )
        )

        loss = positive_loss + negative_loss

        return loss.mean()