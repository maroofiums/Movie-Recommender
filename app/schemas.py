from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    title: str
    top_k: int = 10


class MovieResponse(BaseModel):
    title: str
    score: float
    genres: str
    year: int
    rating: float


class RecommendationResponse(BaseModel):
    query: str
    recommendations: list[MovieResponse]