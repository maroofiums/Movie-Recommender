"""
FastAPI application.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.dependencies import (
    load_models,
)

from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):

    load_models()

    yield


app = FastAPI(

    title="Movie Recommendation API",

    version="1.0",

    lifespan=lifespan,

)

app.include_router(router)