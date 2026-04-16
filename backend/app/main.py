from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.router import router
from backend.app.config import get_settings

app = FastAPI()

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_origins.split(","),
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)