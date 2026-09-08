"""Fetching and parsing: the metadata contract, for both source types.

Every downstream consumer (context assembly, citations, eval labels,
re-ingestion) assumes the same four fields exist on every document. Optional
fields ride along but nothing may depend on them.
"""

import hashlib

import frontmatter
import pytest

from src.fetch_and_parse import (
    RAW,
    PROCESSED,
    fetch_git,
    first_heading,
    process_git,
    process_url,
)
from tests.conftest import long_markdown

REQUIRED = {"doc_id", "title", "url", "doc_hash"}


def read_processed(rel):
    return frontmatter.loads((PROCESSED / rel).read_text(encoding="utf-8"))


def write_raw(rel, text):
    path = RAW / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- title fallback chain -------------------------------------------------

def test_first_heading_finds_h1_only():
    assert first_heading("## Sub\n# Real Title\ntext") == "Real Title"
    assert first_heading("## Sub\nno h1 here") is None


@pytest.mark.parametrize(
    "body, expected",
    [
        ("---\ntitle: From Frontmatter\n---\n# From Heading\ntext", "From Frontmatter"),
        ("# From Heading\ntext", "From Heading"),
        ("no headings at all, just prose", "page"),  # falls back to filename
    ],
)
def test_title_falls_back_frontmatter_then_heading_then_filename(
    workdir, git_source, body, expected
):
    write_raw("field-guide/page.md", body)
    process_git(git_source, "deadbeef")
    assert read_processed("field-guide/page.md")["title"] == expected


# --- the required contract ------------------------------------------------

def test_git_documents_carry_every_required_field(workdir, git_source):
    write_raw("field-guide/a/page.md", long_markdown())
    process_git(git_source, "deadbeef")

    meta = read_processed("field-guide/a/page.md").metadata
    assert REQUIRED <= meta.keys()
    assert meta["doc_id"] == "field-guide/a/page"


def test_url_documents_carry_every_required_field(workdir, html_page):
    source = {"id": "genai-platform", "type": "url", "url": "https://example.com/post"}
    write_raw("genai-platform.html", html_page)
    process_url(source)

    meta = read_processed("genai-platform.md").metadata
    assert REQUIRED <= meta.keys()
    assert meta["doc_id"] == "genai-platform"
    assert meta["url"] == source["url"]


def test_url_keeps_optional_metadata_that_git_sources_lack(workdir, html_page, git_source):
    write_raw("genai-platform.html", html_page)
    process_url({"id": "genai-platform", "type": "url", "url": "https://example.com/post"})

    write_raw("field-guide/page.md", long_markdown())
    process_git(git_source, "deadbeef")

    url_meta = read_processed("genai-platform.md").metadata
    git_meta = read_processed("field-guide/page.md").metadata

    # optional fields are allowed to differ between source types...
    assert "author" in url_meta
    assert "author" not in git_meta
    # ...but the contract holds for both
    assert REQUIRED <= url_meta.keys()
    assert REQUIRED <= git_meta.keys()


def test_url_metadata_never_wins_over_the_manifest(workdir, html_page):
    """The manifest is the source of truth for where a document came from,
    not whatever the page claims about itself."""
    write_raw("genai-platform.html", html_page)
    process_url({"id": "genai-platform", "type": "url", "url": "https://manifest.example/post"})
    assert read_processed("genai-platform.md")["url"] == "https://manifest.example/post"


# --- doc_hash -------------------------------------------------------------

def test_doc_hash_is_sha256_of_the_body_not_the_file(workdir, git_source):
    body = long_markdown()
    write_raw("field-guide/page.md", f"---\ntitle: T\n---\n{body}")
    process_git(git_source, "deadbeef")

    post = read_processed("field-guide/page.md")
    expected = hashlib.sha256(post.content.encode("utf-8")).hexdigest()
    assert post["doc_hash"] == expected


def test_doc_hash_is_stable_when_content_is_unchanged(workdir, git_source):
    write_raw("field-guide/page.md", long_markdown())
    process_git(git_source, "sha-one")
    first = read_processed("field-guide/page.md")["doc_hash"]

    process_git(git_source, "sha-two")  # different commit, same content
    assert read_processed("field-guide/page.md")["doc_hash"] == first


def test_doc_hash_changes_when_content_changes(workdir, git_source):
    write_raw("field-guide/page.md", long_markdown(title="One"))
    process_git(git_source, "deadbeef")
    first = read_processed("field-guide/page.md")["doc_hash"]

    write_raw("field-guide/page.md", long_markdown(title="Two"))
    process_git(git_source, "deadbeef")
    assert read_processed("field-guide/page.md")["doc_hash"] != first


# --- frontmatter handling -------------------------------------------------

def test_source_frontmatter_is_stripped_from_the_body(workdir, git_source):
    write_raw("field-guide/page.md", "---\ntitle: T\ndraft: true\n---\n# Heading\n\nBody.")
    process_git(git_source, "deadbeef")

    content = read_processed("field-guide/page.md").content
    assert "draft" not in content
    assert content.startswith("# Heading")


# --- citation urls --------------------------------------------------------

def test_git_citation_url_pins_the_commit(workdir, git_source):
    write_raw("field-guide/interview/questions.md", long_markdown())
    process_git(git_source, "9f2c1a")

    url = read_processed("field-guide/interview/questions.md")["url"]
    assert "/blob/9f2c1a/" in url
    assert url.endswith("interview/questions.md")
    assert "\\" not in url  # forward slashes, on every platform


# --- fetching from a real repo -------------------------------------------

def test_fetch_git_copies_markdown_and_honours_excludes(workdir, git_source):
    sha = fetch_git(git_source)

    copied = {p.relative_to(RAW / "field-guide").as_posix()
              for p in (RAW / "field-guide").rglob("*.md")}

    assert copied == {"guide/interview.md", "README.md"}  # STYLING.md excluded
    assert len(sha) == 40 and sha.isalnum()
    assert not list((RAW / "field-guide").rglob("*.txt"))  # markdown only
