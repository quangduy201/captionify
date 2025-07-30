from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("🛑 App is shutting down...")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://captionify-app.onrender.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
        "http://localhost:10000",
        "http://127.0.0.1:10000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
