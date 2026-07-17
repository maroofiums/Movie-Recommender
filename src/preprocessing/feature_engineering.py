from __future__ import annotations

import ast
import logging

import pandas as pd

from config.settings import (
    CLEAN_MOVIES_PATH,
    FEATURED_MOVIES_PATH,
)

logger = logging.getLogger(__name__)


class MovieFeatureEngineer:
    """Generate structured movie features."""

    def load_data(self) -> pd.DataFrame:
        """Load cleaned dataset."""

        logger.info("Loading cleaned dataset...")

        return pd.read_csv(
            CLEAN_MOVIES_PATH,
            low_memory=False,
        )

    @staticmethod
    def parse_json_column(value: str):
        """
        Parse JSON-like string safely.

        Returns:
            list | dict
        """

        if pd.isna(value):
            return []

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    def extract_genres(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract genre names."""

        def parser(value):

            genres = self.parse_json_column(value)

            if not isinstance(genres, list):
                return ""

            return " ".join(
                genre["name"]
                for genre in genres
                if "name" in genre
            )

        df["genres_text"] = df["genres"].apply(parser)

        return df

    def extract_collection(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract collection name."""

        def parser(value):

            collection = self.parse_json_column(value)

            if isinstance(collection, dict):
                return collection.get("name", "")

            return ""

        df["collection"] = (
            df["belongs_to_collection"]
            .apply(parser)
        )

        return df

    def extract_companies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract production companies."""

        def parser(value):

            companies = self.parse_json_column(value)

            if not isinstance(companies, list):
                return ""

            return " ".join(
                company["name"]
                for company in companies
                if "name" in company
            )

        df["companies"] = (
            df["production_companies"]
            .apply(parser)
        )

        return df

    def extract_release_year(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract release year."""

        df["release_date"] = pd.to_datetime(
            df["release_date"],
            errors="coerce",
        )

        df["release_year"] = (
            df["release_date"]
            .dt.year
            .fillna(0)
            .astype(int)
        )

        return df

    def runtime_category(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create runtime category."""

        def category(runtime):

            if pd.isna(runtime):
                return "Unknown"

            if runtime < 60:
                return "Short"

            if runtime < 120:
                return "Medium"

            return "Long"

        df["runtime_category"] = (
            df["runtime"]
            .apply(category)
        )

        return df

    def save(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Save processed dataset."""

        FEATURED_MOVIES_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            FEATURED_MOVIES_PATH,
            index=False,
        )

        logger.info(
            "Saved engineered dataset to %s",
            FEATURED_MOVIES_PATH,
        )

    def run(self) -> pd.DataFrame:
        """Execute feature engineering."""

        logger.info("Starting feature engineering...")

        df = self.load_data()

        df = self.extract_genres(df)

        df = self.extract_collection(df)

        df = self.extract_companies(df)

        df = self.extract_release_year(df)

        df = self.runtime_category(df)

        self.save(df)

        logger.info("Feature engineering completed.")

        return df