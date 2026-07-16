"""
Data cleaning module for the Movie Recommendation Engine.

Responsibilities:
- Load raw dataset
- Validate required columns
- Remove duplicates
- Handle missing values
- Convert data types
- Save cleaned dataset
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import (
    CLEAN_MOVIES_PATH,
    MOVIES_METADATA_PATH,
)

logger = logging.getLogger(__name__)


class MovieDataCleaner:
    """Clean and preprocess the raw movie metadata dataset."""

    REQUIRED_COLUMNS = [
        "id",
        "title",
        "overview",
        "genres",
        "tagline",
        "original_language",
        "release_date",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "belongs_to_collection",
        "production_companies",
    ]

    def __init__(self, dataset_path: Path = MOVIES_METADATA_PATH):
        self.dataset_path = dataset_path

    def load_data(self) -> pd.DataFrame:
        """
        Load the raw dataset.

        Returns:
            Loaded DataFrame.
        """
        logger.info("Loading dataset from %s", self.dataset_path)

        return pd.read_csv(
            self.dataset_path,
            low_memory=False,
        )

    def validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate required columns exist.

        Args:
            df: Input DataFrame.

        Raises:
            ValueError: Missing required columns.
        """

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def remove_duplicates(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove duplicate movies.

        Args:
            df: Input DataFrame.

        Returns:
            Clean DataFrame.
        """

        before = len(df)

        df = df.drop_duplicates(
            subset=["id"],
            keep="first",
        )

        logger.info(
            "Removed %d duplicate rows.",
            before - len(df),
        )

        return df

    def clean_missing_values(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fill missing values.

        Args:
            df: Input DataFrame.

        Returns:
            Clean DataFrame.
        """

        df["overview"] = df["overview"].fillna("")
        df["tagline"] = df["tagline"].fillna("")
        df["genres"] = df["genres"].fillna("[]")
        df["belongs_to_collection"] = (
            df["belongs_to_collection"]
            .fillna("{}")
        )
        df["production_companies"] = (
            df["production_companies"]
            .fillna("[]")
        )

        return df

    def convert_types(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert columns to proper data types.

        Args:
            df: Input DataFrame.

        Returns:
            Converted DataFrame.
        """

        df["release_date"] = pd.to_datetime(
            df["release_date"],
            errors="coerce",
        )

        numeric_columns = [
            "runtime",
            "vote_average",
            "vote_count",
            "popularity",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        return df

    def save_data(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Save cleaned dataset.

        Args:
            df: Clean DataFrame.
        """

        CLEAN_MOVIES_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            CLEAN_MOVIES_PATH,
            index=False,
        )

        logger.info(
            "Clean dataset saved to %s",
            CLEAN_MOVIES_PATH,
        )

    def run(self) -> pd.DataFrame:
        """
        Execute the complete cleaning pipeline.

        Returns:
            Clean DataFrame.
        """

        logger.info("Starting data cleaning pipeline...")

        df = self.load_data()

        self.validate_columns(df)

        df = self.remove_duplicates(df)

        df = self.clean_missing_values(df)

        df = self.convert_types(df)

        self.save_data(df)

        logger.info("Data cleaning completed successfully.")

        return df