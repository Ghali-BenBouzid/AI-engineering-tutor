from langchain_core.documents import Document

from src.index import get_store
from src.embed import get_embeddings
from src.config import settings

def top_k_search(query: str, top_k=settings.top_k) -> list[tuple[Document, float]]:
    vector_store = get_store()
    embeddings = get_embeddings()

    if embeddings.model_name != vector_store._collection_metadata["embedding_model"]:
        raise RuntimeError(f"The retrieval embedding model {embeddings.model_name} does not match the one used for indexation {vector_store._collection_metadata["embedding_model"]}")

    results = vector_store.similarity_search_by_vector_with_relevance_scores(
        embedding=embeddings.embed_query(query),
        k=top_k,
    )

    return results


def should_abstain(hits: list[tuple[Document, float]]) -> bool:
    """The guardrail. Chroma returns cosine distance here, so smaller is closer.

    Lives here rather than in answer.py so the eval measures the rule itself
    instead of a copy of it that can drift.
    """
    return not hits or hits[0][1] > settings.max_distance


def main() -> None:
    user_query = "What is query rewriting ?"

    retrieved_docs = top_k_search(
        query=user_query,
    )

    for doc, score in retrieved_docs:
        print(f"Retrieved chunk | score = {score}")
        print(doc.page_content)
        print("Chunk metadata: ")
        print(doc.metadata)


if __name__=="__main__":
    main()