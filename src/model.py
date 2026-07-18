from __future__ import annotations

import torch
import torch.nn as nn

from transformers import AutoModel

from config.settings import (
    MODEL_NAME,
    EMBEDDING_DIM,
)


class MeanPooling(nn.Module):
    """
    Mean Pooling over token embeddings.
    """

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        mask = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        )

        mask = mask.float()

        summed = torch.sum(
            last_hidden_state * mask,
            dim=1,
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9,
        )

        return summed / counts


class MovieEncoder(nn.Module):
    """
    Transformer encoder for movie text.
    """

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.pooling = MeanPooling()

        self.projection = nn.Sequential(
            nn.Linear(
                EMBEDDING_DIM,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                256,
                128,
            ),
        )

        self.normalize = nn.functional.normalize

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        embedding = self.pooling(
            outputs.last_hidden_state,
            attention_mask,
        )

        embedding = self.projection(
            embedding
        )

        embedding = self.normalize(
            embedding,
            p=2,
            dim=1,
        )

        return embedding


class SiameseNetwork(nn.Module):
    """
    Siamese Transformer.
    """

    def __init__(self):

        super().__init__()

        self.encoder = MovieEncoder()

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
    ):

        embedding_a = self.encoder(
            input_ids_a,
            attention_mask_a,
        )

        embedding_b = self.encoder(
            input_ids_b,
            attention_mask_b,
        )

        return embedding_a, embedding_b