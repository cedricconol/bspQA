from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    qdrant_cluster_endpoint: str
    qdrant_api_key: str
    qdrant_collection_name: str


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Returns:
        The application Settings instance, constructed once on first call.
    """
    return Settings()
