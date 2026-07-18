from __future__ import annotations

from difflib import get_close_matches

import pandas as pd


class MovieSearch:
    """
    Search movies by title.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        self.df = dataframe

        self.title_map = {
            title.lower(): idx
            for idx, title in enumerate(
                self.df["title"]
            )
        }

    def exact_search(
        self,
        title: str,
    ) -> pd.Series | None:
        """
        Exact movie search.
        """

        idx = self.title_map.get(
            title.lower()
        )

        if idx is None:
            return None

        return self.df.iloc[idx]

    def fuzzy_search(
        self,
        title: str,
    ) -> pd.Series | None:
        """
        Fuzzy title search.
        """

        matches = get_close_matches(
            title.lower(),
            self.title_map.keys(),
            n=1,
            cutoff=0.6,
        )

        if not matches:
            return None

        idx = self.title_map[
            matches[0]
        ]

        return self.df.iloc[idx]

    def search(
        self,
        title: str,
    ) -> pd.Series | None:
        """
        Search movie by title.

        Exact search first,
        fuzzy search otherwise.
        """

        movie = self.exact_search(
            title
        )

        if movie is not None:
            return movie

        return self.fuzzy_search(
            title
        )