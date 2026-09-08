"""Chunking invariants.

The ceiling is read from the tokenizer, never written down here: if the
embedding model changes, these tests follow it.
"""

import frontmatter
import pytest

from src.chunk import (
    CARRIED,
    PROCESSED,
    chunk_all,
    chunk_doc,
    token_len,
    tokenizer,
)
from src.config import settings
from tests.conftest import long_markdown

MAX_TOKENS = tokenizer.model_max_length


def write_processed(rel, meta, body):
    path = PROCESSED / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.dumps(frontmatter.Post(body, **meta)), encoding="utf-8")


def doc_meta(doc_id="a-doc", **extra):
    return {
        "doc_id": doc_id,
        "title": "A Doc",
        "url": f"https://example.com/{doc_id}",
        "doc_hash": "0" * 64,
        **extra,
    }


# --- the ceiling ----------------------------------------------------------

def test_no_chunk_exceeds_the_models_sequence_length():
    chunks = chunk_doc(doc_meta(), long_markdown(sections=6, sentences=120))
    assert chunks
    oversized = [c for c in chunks if token_len(c["text"]) > MAX_TOKENS]
    assert not oversized, f"{len(oversized)} chunks over {MAX_TOKENS} tokens"


def test_chunk_size_is_measured_in_tokens_not_characters():
    """The classic mistake: RecursiveCharacterTextSplitter counts characters
    by default, so `chunk_size=450` silently means ~450 chars (~100 tokens)."""
    chunks = chunk_doc(doc_meta(), long_markdown(sections=2, sentences=200))
    longest = max(token_len(c["text"]) for c in chunks)

    assert longest > settings.chunk_size / 2, (
        f"longest chunk is {longest} tokens; if the splitter were counting "
        f"characters this would sit near {settings.chunk_size // 4}"
    )
    assert longest <= MAX_TOKENS


def test_splitter_uses_the_configured_size():
    """chunk_size/chunk_overlap live in config; the splitter must read them."""
    from src.chunk import size_splitter

    assert size_splitter._chunk_size == settings.chunk_size
    assert size_splitter._chunk_overlap == settings.chunk_overlap


# --- ids ------------------------------------------------------------------

def test_ids_are_deterministic_across_runs():
    meta, body = doc_meta(), long_markdown()
    assert chunk_doc(meta, body) == chunk_doc(meta, body)


def test_ids_are_unique_and_namespaced_by_document(workdir):
    write_processed("one.md", doc_meta("one"), long_markdown("One"))
    write_processed("nested/two.md", doc_meta("nested/two"), long_markdown("Two"))

    chunks = chunk_all()
    ids = [c["id"] for c in chunks]

    assert len(ids) == len(set(ids))
    assert all(c["id"].startswith(c["doc_id"] + ":") for c in chunks)


def test_ids_are_contiguous_within_a_document():
    chunks = chunk_doc(doc_meta("d"), long_markdown())
    assert [c["id"] for c in chunks] == [f"d:{i}" for i in range(len(chunks))]


# --- metadata propagation -------------------------------------------------

def test_every_chunk_carries_the_required_document_fields():
    meta = doc_meta()
    for chunk in chunk_doc(meta, long_markdown()):
        for field in ("doc_id", "title", "url", "doc_hash"):
            assert chunk[field] == meta[field]


@pytest.mark.parametrize("field", CARRIED)
def test_optional_fields_propagate_when_present(field):
    chunks = chunk_doc(doc_meta(**{field: "value"}), long_markdown())
    assert all(c[field] == "value" for c in chunks)


@pytest.mark.parametrize("field", CARRIED)
def test_optional_fields_are_absent_not_none_when_missing(field):
    """Absent, never None -- Chroma rejects None metadata values."""
    for chunk in chunk_doc(doc_meta(), long_markdown()):
        assert field not in chunk


def test_document_fields_nobody_asked_for_do_not_leak_into_chunks():
    meta = doc_meta(description="a long summary repeated on every chunk",
                    hostname="example.com", sitename="Example")
    for chunk in chunk_doc(meta, long_markdown()):
        assert "description" not in chunk
        assert "hostname" not in chunk
        assert "sitename" not in chunk


def test_no_chunk_metadata_value_is_none():
    for chunk in chunk_doc(doc_meta(author=None), long_markdown()):
        assert None not in chunk.values()


# --- sections -------------------------------------------------------------

def test_section_path_is_the_nested_heading_trail():
    chunks = chunk_doc(doc_meta(), long_markdown(title="Guide", sections=1))
    paths = {c["section"] for c in chunks}

    assert any(p == "Guide > Section 0" for p in paths)
    assert any(p == "Guide > Section 0 > Sub 0" for p in paths)


def test_headings_stay_in_the_chunk_text():
    """strip_headers=False: the heading words are the most semantically
    loaded text in a section, so the embedding has to see them."""
    chunks = chunk_doc(doc_meta(), long_markdown(sections=1))
    assert any("## Section 0" in c["text"] for c in chunks)


def test_a_document_with_no_subheadings_still_chunks():
    chunks = chunk_doc(doc_meta(), "# Only A Title\n\n" + "Prose. " * 800)
    assert chunks
    assert all(c["section"] == "Only A Title" for c in chunks)


# --- corpus level ---------------------------------------------------------

def test_chunk_all_is_empty_on_an_empty_corpus(workdir):
    assert chunk_all() == []


def test_chunk_all_covers_every_processed_document(workdir):
    write_processed("one.md", doc_meta("one"), long_markdown("One"))
    write_processed("deep/nested/two.md", doc_meta("deep/nested/two"), long_markdown("Two"))

    assert {c["doc_id"] for c in chunk_all()} == {"one", "deep/nested/two"}
