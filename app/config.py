from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Main OpenAI endpoint — used for LLM chat AND embeddings.
    openai_api_base_url: str | None = None
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1024

    # Separate image generation endpoint (different provider/proxy/key).
    openai_image_base_url: str | None = None
    openai_image_api_key: str = ""
    openai_image_model: str = "gpt-image-1"

    # ChromaDB
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "kicau_rag"

    # Laravel / Supabase storage endpoint
    laravel_storage_url: str = ""
    laravel_storage_token: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
