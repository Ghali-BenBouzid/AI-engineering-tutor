from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 450
    chunk_overlap: int = 90
    top_k: int = 5
    max_distance: float = 0.4 # maximum tolerated distance between the embedded user query and the embedded chunks at retrieval

    llm_model: str = "meta-llama/llama-3.3-70b-instruct"

    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    lockfile_path: str = "data/sources.lock.yml"

    chroma_dir: str = "data/chroma"
    collection_name: str = "ai_engineering_collection"

    openrouter_api_key: str | None = None

settings = Settings()