"""The embedder is the one piece of code on both the offline and online paths.

These tests exist because every failure here is silent: wrong vectors still
rank, still return five results, and never raise.
"""

import math

import pytest

from src.chunk import tokenizer
from src.config import settings
from src.embed import get_embeddings

pytestmark = pytest.mark.slow

PASSAGE = "Query rewriting reformulates the user question before retrieval."
QUESTION = "what is query rewriting?"


@pytest.fixture(scope="module")
def embeddings():
    return get_embeddings()


def norm(vector):
    return math.sqrt(sum(x * x for x in vector))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_document_vectors_are_unit_length(embeddings):
    for vector in embeddings.embed_documents([PASSAGE, "another passage"]):
        assert norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_query_vectors_are_unit_length(embeddings):
    """query_encode_kwargs *replaces* encode_kwargs rather than merging, so
    normalize_embeddings has to be repeated there or queries come out unscaled."""
    assert norm(embeddings.embed_query(QUESTION)) == pytest.approx(1.0, abs=1e-5)


def test_dot_product_equals_cosine_on_normalised_vectors(embeddings):
    q = embeddings.embed_query(QUESTION)
    d = embeddings.embed_documents([PASSAGE])[0]
    cosine = dot(q, d) / (norm(q) * norm(d))
    assert dot(q, d) == pytest.approx(cosine, abs=1e-6)


def test_the_query_path_is_not_the_document_path(embeddings):
    """bge prepends an instruction to queries only. Same text through the two
    methods must therefore produce different vectors -- if they match, the
    prefix was dropped and retrieval is running symmetric."""
    as_query = embeddings.embed_query(PASSAGE)
    as_document = embeddings.embed_documents([PASSAGE])[0]
    assert dot(as_query, as_document) < 0.999


def test_the_instruction_is_only_on_the_query_side(embeddings):
    """Guards the mirror-image bug: prefixing passages too."""
    prefix = embeddings.query_encode_kwargs["prompt"]
    plain = embeddings.embed_documents([PASSAGE])[0]
    prefixed = embeddings.embed_documents([prefix + PASSAGE])[0]
    assert dot(plain, prefixed) < 0.999


def test_relevant_passages_outrank_irrelevant_ones(embeddings):
    q = embeddings.embed_query(QUESTION)
    relevant, irrelevant = embeddings.embed_documents(
        [PASSAGE, "Pizza dough hydration is the ratio of water to flour."]
    )
    assert dot(q, relevant) > dot(q, irrelevant)


def test_embedding_dimension_is_consistent_across_both_paths(embeddings):
    assert len(embeddings.embed_query(QUESTION)) == len(
        embeddings.embed_documents([PASSAGE])[0]
    )


def test_the_tokenizer_matches_the_embedding_model(embeddings):
    """chunk.py sizes chunks with this tokenizer; embed.py encodes with that
    model. If they ever drift, chunks are sized against the wrong ceiling."""
    assert tokenizer.name_or_path == settings.embedding_model_name
    assert embeddings.model_name == settings.embedding_model_name
