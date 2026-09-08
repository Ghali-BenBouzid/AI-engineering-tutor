"""Indexing against a real Chroma collection in a tmp directory.

The two properties that matter: re-indexing must not duplicate, and the
collection must record which model built it.
"""

import frontmatter
import pytest

from src.config import settings
from src.index import get_store, index_chunks
from tests.conftest import long_markdown

pytestmark = pytest.mark.slow


@pytest.fixture
def store(workdir, monkeypatch):
    """Chroma caches clients by persist_directory, and settings.chroma_dir is a
    relative string -- so every test must get its own absolute path or the
    second one reuses a client pointing at a deleted tmp dir."""
    import chromadb

    monkeypatch.setattr(settings, "chroma_dir", str(workdir / "chroma"))
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    get_store.cache_clear()

    yield get_store()

    get_store.cache_clear()
    chromadb.api.client.SharedSystemClient.clear_system_cache()


def small_corpus(workdir):
    from src.chunk import PROCESSED, chunk_all

    meta = {
        "doc_id": "a-doc",
        "title": "A Doc",
        "url": "https://example.com/a-doc",
        "doc_hash": "0" * 64,
        "author": "Someone",
    }
    path = PROCESSED / "a-doc.md"
    path.write_text(
        frontmatter.dumps(frontmatter.Post(long_markdown(sections=1), **meta)),
        encoding="utf-8",
    )
    return chunk_all()


def test_indexing_stores_every_chunk(store, workdir):
    chunks = small_corpus(workdir)
    index_chunks(chunks)
    assert store._collection.count() == len(chunks)


def test_reindexing_upserts_instead_of_duplicating(store, workdir):
    """Deterministic ids are what make ingestion re-runnable. Without them
    Chroma assigns UUIDs and a second run doubles the corpus."""
    chunks = small_corpus(workdir)
    index_chunks(chunks)
    index_chunks(chunks)
    assert store._collection.count() == len(chunks)


def test_collection_records_the_model_that_built_it(store, workdir):
    index_chunks(small_corpus(workdir))
    metadata = store._collection.metadata
    assert metadata["embedding_model"] == settings.embedding_model_name


def test_collection_uses_cosine_distance(store, workdir):
    """Scores are calibrated as cosine similarity. Chroma defaults to L2,
    where the comparison runs the other way and the threshold inverts."""
    index_chunks(small_corpus(workdir))
    assert store._collection.metadata["hnsw:space"] == "cosine"


def test_stored_metadata_matches_the_chunk_record(store, workdir):
    chunks = small_corpus(workdir)
    index_chunks(chunks)

    stored = store._collection.get(ids=[chunks[0]["id"]], include=["metadatas", "documents"])
    metadata = stored["metadatas"][0]

    assert stored["documents"][0] == chunks[0]["text"]
    for field in ("doc_id", "title", "url", "doc_hash", "section", "author"):
        assert metadata[field] == chunks[0][field]
    assert "text" not in metadata     # the body is the document, not metadata
    assert "id" not in metadata       # the id is the key, not metadata


def test_similarity_search_returns_indexed_chunks(store, workdir):
    index_chunks(small_corpus(workdir))
    hits = store.similarity_search_with_relevance_scores("retrieval and embeddings", k=3)

    assert hits
    assert all(0.0 <= score <= 1.0 for _, score in hits)
    assert all("doc_id" in doc.metadata for doc, _ in hits)
