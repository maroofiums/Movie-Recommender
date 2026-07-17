from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    FEATURED_MOVIES_PATH,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)


class MovieTextBuilder:
    """Create textual representation for every movie."""

    OUTPUT_PATH = PROCESSED_DATA_DIR / "movies_with_text.csv"

    def load_data(self) -> pd.DataFrame:
        """Load engineered dataset."""

        logger.info("Loading engineered dataset...")

        return pd.read_csv(
            FEATURED_MOVIES_PATH,
            low_memory=False,
        )

    @staticmethod
    def clean_text(value: object) -> str:
        """
        Normalize text values.

        Args:
            value: Input value.

        Returns:
            Clean string.
        """

        if pd.isna(value):
            return ""

        return (
            str(value)
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )

    def build_movie_text(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build a rich textual representation.

        Returns:
            DataFrame with movie_text column.
        """

        texts = []

        for _, row in df.iterrows():

            text = f"""
                Title: {self.clean_text(row["title"])}

                Genres: {self.clean_text(row["genres_text"])}

                Overview:
                {self.clean_text(row["overview"])}

                Tagline:
                {self.clean_text(row["tagline"])}

                Collection:
                {self.clean_text(row["collection"])}

                Production Companies:
                {self.clean_text(row["companies"])}

                Language:
                {self.clean_text(row["original_language"])}

                Runtime:
                {self.clean_text(row["runtime_category"])}

                Release Year:
                {self.clean_text(row["release_year"])}
            """

            texts.append(" ".join(text.split()))

        df["movie_text"] = texts

        return df

    def save(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Save processed dataset."""

        self.OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            self.OUTPUT_PATH,
            index=False,
        )

        logger.info(
            "Saved dataset to %s",
            self.OUTPUT_PATH,
        )

    def run(self) -> pd.DataFrame:
        """Execute text generation pipeline."""

        logger.info("Building movie text...")

        df = self.load_data()

        df = self.build_movie_text(df)

        self.save(df)

        logger.info("Movie text generation completed.")

        return df