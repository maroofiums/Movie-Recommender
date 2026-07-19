from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
)

from src.inference import MovieInference
from src.retrieval.faiss_index import MovieIndex
from src.retrieval.recommender import MovieRecommender
from src.retrieval.search import MovieSearch


class AppState:

    recommender: MovieRecommender | None = None

    search: MovieSearch | None = None


state = AppState()


def load_models():

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    movies = pd.read_parquet(
        Path("models/movies.parquet")
    )

    index = MovieIndex()

    index.load(
        FAISS_INDEX_PATH
    )

    index.load_movies(
        movies
    )

    inference = MovieInference(
        checkpoint_path=Path(
            "models/best_model.pt"
        )
    )

    state.recommender = MovieRecommender(
        inference=inference,
        index=index,
    )

    state.search = MovieSearch(
        movies
    )