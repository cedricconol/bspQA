from fastapi import FastAPI

from backend.app.api.routes.router import router

app = FastAPI()
app.include_router(router)