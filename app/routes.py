from fastapi import APIRouter, HTTPException

from app.dependencies import state
from app.schemas import (
    RecommendationRequest,
    RecommendationResponse,
)

router = APIRouter()


@router.get("/health")
def health():

    return {
        "status": "ok"
    }


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
):

    movie = state.search.search(
        request.title
    )

    if movie is None:

        raise HTTPException(
            status_code=404,
            detail="Movie not found",
        )

    recommendations = (
        state.recommender.recommend(
            movie["movie_text"],
            request.top_k,
        )
    )

    return RecommendationResponse(
        query=request.title,
        recommendations=recommendations,
    )