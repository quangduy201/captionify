from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("🛑 App is shutting down...")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)
