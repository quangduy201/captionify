from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router
from app.model.load_model import load_model_and_vocab_from_kaggle


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except IOError:
        print("🟡 Downloading en_core_web_sm...")
        from spacy.cli import download
        download("en_core_web_sm")

    load_model_and_vocab_from_kaggle()
    yield
    print("🛑 App is shutting down...")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)
