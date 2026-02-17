"""
Configuration management for the Research Paper RAG System.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"

    # Vector Store
    vector_store_type: Literal["chroma", "faiss"] = "chroma"
    chroma_persist_dir: Path = Path("./data/chroma_db")

    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_tokens: int = 8000

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    @property
    def papers_dir(self) -> Path:
        return Path("./data/papers")

    @property
    def processed_dir(self) -> Path:
        return Path("./data/processed")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings

