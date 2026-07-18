from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config.settings import (
    FEATURED_MOVIES_PATH,
    MODEL_NAME,
    MAX_SEQUENCE_LENGTH,
)

from src.model import MovieEncoder


class MovieInference:
    """Generate movie embeddings using the trained encoder."""

    def __init__(
        self,
        checkpoint_path,
        device=None,
    ):

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = MovieEncoder().to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )

        self.model.eval()

    def tokenize(self, text: str):

        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )

    @torch.no_grad()
    def encode(self, text: str):

        tokens = self.tokenize(text)

        embedding = self.model(

            tokens["input_ids"].to(self.device),

            tokens["attention_mask"].to(self.device),

        )

        return embedding.squeeze().cpu().numpy()

    @torch.no_grad()
    def build_embeddings(self):

        df = pd.read_csv(
            FEATURED_MOVIES_PATH,
        )

        embeddings = []

        for text in tqdm(df["movie_text"]):

            embeddings.append(
                self.encode(text)
            )

        embeddings = np.vstack(
            embeddings
        )

        return df, embeddings