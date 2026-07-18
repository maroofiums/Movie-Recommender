from __future__ import annotations

import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config.settings import (
    MODEL_NAME,
    MAX_SEQUENCE_LENGTH,
)


class MoviePairDataset(Dataset):
    """Dynamic movie pair dataset."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

    def __len__(self):

        return len(self.df)

    def split_genres(
        self,
        genres: str,
    ) -> set:

        if pd.isna(genres):
            return set()

        return set(genres.split())

    def is_positive(
        self,
        idx1: int,
        idx2: int,
    ) -> bool:

        movie1 = self.df.iloc[idx1]
        movie2 = self.df.iloc[idx2]

        if (
            movie1["collection"]
            and movie1["collection"] == movie2["collection"]
        ):
            return True

        genres1 = self.split_genres(
            movie1["genres_text"]
        )

        genres2 = self.split_genres(
            movie2["genres_text"]
        )

        return len(genres1 & genres2) >= 2

    def sample_pair(
        self,
        index: int,
    ):

        anchor = self.df.iloc[index]

        positive = random.random() < 0.5

        while True:

            candidate_index = random.randint(
                0,
                len(self.df) - 1,
            )

            if candidate_index == index:
                continue

            same = self.is_positive(
                index,
                candidate_index,
            )

            if same == positive:
                break

        return (
            anchor,
            self.df.iloc[candidate_index],
            int(positive),
        )

    def tokenize(
        self,
        text: str,
    ):

        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )

    def __getitem__(
        self,
        index: int,
    ):

        movie_a, movie_b, label = self.sample_pair(
            index
        )

        token_a = self.tokenize(
            movie_a["movie_text"]
        )

        token_b = self.tokenize(
            movie_b["movie_text"]
        )

        return {
            "input_ids_a": token_a["input_ids"].squeeze(),
            "attention_mask_a": token_a["attention_mask"].squeeze(),
            "input_ids_b": token_b["input_ids"].squeeze(),
            "attention_mask_b": token_b["attention_mask"].squeeze(),
            "label": torch.tensor(
                label,
                dtype=torch.float,
            ),
        }