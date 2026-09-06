from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 450
    chunk_overlap: int = 90
    top_k: int = 5

    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    lockfile_path: str = "data/sources.lock.yml"

    chroma_dir: str = "data/chroma"
    collection_name: str = "ai_engineering_collection"

    openrouter_key: str | None = None

settings = Settings()