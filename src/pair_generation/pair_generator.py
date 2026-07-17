from __future__ import annotations

import itertools
import logging

import pandas as pd

from config.settings import (
    PAIR_DATASET_PATH,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)


class PairGenerator:

    INPUT_PATH = PROCESSED_DATA_DIR / "movies_with_text.csv"

    def load_data(self) -> pd.DataFrame:
        logger.info("Loading processed movies...")

        return pd.read_csv(self.INPUT_PATH)

    @staticmethod
    def split_genres(genres: str) -> set[str]:

        if pd.isna(genres):
            return set()

        return set(genres.split())

    def positive_pair(self, movie1, movie2) -> bool:

        # Same collection
        if (
            movie1["collection"]
            and movie1["collection"] == movie2["collection"]
        ):
            return True

        genres1 = self.split_genres(movie1["genres_text"])
        genres2 = self.split_genres(movie2["genres_text"])

        common = genres1 & genres2

        if (
            len(common) >= 2
            and movie1["original_language"] == movie2["original_language"]
        ):
            return True

        return False

    def negative_pair(self, movie1, movie2) -> bool:

        genres1 = self.split_genres(movie1["genres_text"])
        genres2 = self.split_genres(movie2["genres_text"])

        common = genres1 & genres2

        if len(common) > 0:
            return False

        if movie1["collection"] == movie2["collection"]:
            return False

        return True

    def generate_pairs(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Generating training pairs...")

        pairs = []

        combinations = itertools.combinations(
            df.to_dict("records"),
            2,
        )

        for movie1, movie2 in combinations:

            if self.positive_pair(movie1, movie2):

                pairs.append(
                    {
                        "movie_a": movie1["movie_text"],
                        "movie_b": movie2["movie_text"],
                        "label": 1,
                    }
                )

            elif self.negative_pair(movie1, movie2):

                pairs.append(
                    {
                        "movie_a": movie1["movie_text"],
                        "movie_b": movie2["movie_text"],
                        "label": 0,
                    }
                )

        logger.info("Generated %d pairs.", len(pairs))

        return pd.DataFrame(pairs)

    def save(self, df: pd.DataFrame):

        PAIR_DATASET_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            PAIR_DATASET_PATH,
            index=False,
        )

        logger.info("Saved pairs dataset.")

    def run(self):

        df = self.load_data()

        pair_df = self.generate_pairs(df)

        self.save(pair_df)

        return pair_df