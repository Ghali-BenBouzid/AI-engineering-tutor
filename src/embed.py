from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache

from src.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"prompt": "Represent this sentence for searching relevant passages: ", "normalize_embeddings": True}
    )
