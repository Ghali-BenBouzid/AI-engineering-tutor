from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.chunk import chunk_all
from src.config import settings
from src.embed import get_embeddings


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": settings.embedding_model_name,
        },
    )


def index_chunks(chunks: list[dict[str, Any]]) -> None:
    store = get_store()

    docs = [
        Document(
            page_content=c["text"],
            metadata={k: v for k, v in c.items()
                      if k not in ("text", "id") and v is not None},
        )
        for c in chunks
    ]
    store.add_documents(docs, ids=[c["id"] for c in chunks])


def main() -> None:
    chunks = chunk_all()
    index_chunks(chunks)
    print(f"indexed {len(chunks)} chunks -> "
          f"{get_store()._collection.count()} in collection")


if __name__ == "__main__":
    main()