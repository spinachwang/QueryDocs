from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import qa
from src.api.models import HealthResponse
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="QueryDocs API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa.router)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


@app.get("/")
async def root():
    return {"message": "QueryDocs API is running"}