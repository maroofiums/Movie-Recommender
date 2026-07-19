from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config.settings import (
    FEATURED_MOVIES_PATH,
    MODEL_NAME,
    MAX_SEQUENCE_LENGTH,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
)

from src.model import MovieEncoder
from src.retrieval.faiss_index import MovieIndex


class MovieEmbedder:
    """
    Generate embeddings for all movies.
    """

    def __init__(
        self,
        checkpoint_path: Path,
    ) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = MovieEncoder().to(
            self.device
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )

        self.model.eval()

    def tokenize(
        self,
        text: str,
    ):

        return self.tokenizer(
            text,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    @torch.no_grad()
    def encode(
        self,
        text: str,
    ) -> np.ndarray:

        tokens = self.tokenize(text)

        embedding = self.model(

            tokens["input_ids"].to(self.device),

            tokens["attention_mask"].to(self.device),

        )

        return (
            embedding.squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    @torch.no_grad()
    def build(self):

        df = pd.read_csv(
            FEATURED_MOVIES_PATH
        )

        embeddings = []

        for text in tqdm(
            df["movie_text"],
            desc="Encoding Movies",
        ):

            embeddings.append(
                self.encode(text)
            )

        embeddings = np.vstack(
            embeddings
        )

        EMBEDDINGS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            EMBEDDINGS_PATH,
            embeddings,
        )

        metadata_path = (
            EMBEDDINGS_PATH.parent
            / "movies.parquet"
        )

        df.to_parquet(
            metadata_path,
            index=False,
        )

        index = MovieIndex()

        index.build(
            embeddings
        )

        index.save(
            FAISS_INDEX_PATH
        )

        print(
            f"Saved {len(df)} embeddings."
        )