from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router
from app.model.load_model import load_model_and_vocab_from_kaggle
from app.utils.spacy_utils import get_spacy_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_spacy_model()
    load_model_and_vocab_from_kaggle()
    yield
    print("🛑 App is shutting down...")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)
