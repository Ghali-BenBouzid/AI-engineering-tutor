"""End-to-end ingestion: sources.yml -> raw -> processed -> chunks -> index.

Both source types run for real. The URL source is served by a stdlib HTTP
server on localhost, so httpx and trafilatura do their actual work with no
network and nothing mocked.
"""

import importlib
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import frontmatter
import pytest
import yaml

import ingest
import src.fetch_and_parse as fp
from tests.conftest import long_markdown
from src.chunk import PROCESSED, chunk_all
from src.config import settings
from src.index import get_store

pytestmark = pytest.mark.slow


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def served_html(tmp_path_factory, html_page):
    """A real HTTP origin on localhost."""
    root = tmp_path_factory.mktemp("www")
    (root / "post.html").write_text(html_page, encoding="utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/post.html"
    server.shutdown()


@pytest.fixture
def corpus(workdir, monkeypatch, git_source, served_html):
    """A tmp project with one git source and one url source."""
    import chromadb

    monkeypatch.setattr(settings, "chroma_dir", str(workdir / "chroma"))
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    get_store.cache_clear()

    sources = [
        git_source,
        {"id": "genai-platform", "type": "url", "url": served_html},
    ]
    (workdir / "data" / "sources.yml").write_text(
        yaml.safe_dump({"sources": sources}), encoding="utf-8"
    )

    yield sources

    get_store.cache_clear()
    chromadb.api.client.SharedSystemClient.clear_system_cache()


def lockfile():
    return yaml.safe_load(fp.LOCK.read_text(encoding="utf-8"))


# --- the happy path -------------------------------------------------------

def test_ingest_builds_an_index_from_scratch(corpus):
    ingest.ingest()

    chunks = chunk_all()
    assert chunks
    assert get_store()._collection.count() == len(chunks)


def test_ingest_produces_raw_and_processed_for_both_source_types(corpus):
    ingest.ingest()

    assert (fp.RAW / "genai-platform.html").exists()
    assert list((fp.RAW / "field-guide").rglob("*.md"))

    processed = {p.relative_to(PROCESSED).as_posix() for p in PROCESSED.rglob("*.md")}
    assert "genai-platform.md" in processed
    assert any(p.startswith("field-guide/") for p in processed)


def test_every_processed_document_satisfies_the_metadata_contract(corpus):
    ingest.ingest()

    required = {"doc_id", "title", "url", "doc_hash"}
    for path in PROCESSED.rglob("*.md"):
        meta = frontmatter.load(path).metadata
        assert required <= meta.keys(), f"{path} is missing {required - meta.keys()}"


def test_every_indexed_chunk_traces_back_to_a_processed_document(corpus):
    """Nothing may be retrievable that isn't in the corpus."""
    ingest.ingest()

    stored = get_store()._collection.get(include=["metadatas"])
    indexed_docs = {m["doc_id"] for m in stored["metadatas"]}
    on_disk = {frontmatter.load(p)["doc_id"] for p in PROCESSED.rglob("*.md")}

    assert indexed_docs <= on_disk


def test_excludes_survive_the_whole_pipeline(corpus):
    """STYLING.md is excluded in sources.yml; it must not be retrievable."""
    ingest.ingest()

    stored = get_store()._collection.get(include=["metadatas"])
    assert not any("STYLING" in m["doc_id"] for m in stored["metadatas"])


# --- the lockfile ---------------------------------------------------------

def test_lockfile_versions_each_source_by_its_own_scheme(corpus):
    ingest.ingest()
    lock = lockfile()

    assert set(lock) == {"field-guide", "genai-platform"}
    assert len(lock["field-guide"]["commit"]) == 40
    assert lock["field-guide"]["files"] == 2
    assert len(lock["genai-platform"]["sha256"]) == 64
    assert all("fetched_at" in entry for entry in lock.values())


def test_a_second_pass_over_unchanged_sources_changes_nothing(corpus):
    """The lockfile diff is the signal that the corpus moved. An unchanged
    re-run must produce byte-identical output."""
    ingest.ingest()
    first_lock = fp.LOCK.read_text(encoding="utf-8")
    first_count = get_store()._collection.count()

    ingest.ingest()

    assert fp.LOCK.read_text(encoding="utf-8") == first_lock
    assert get_store()._collection.count() == first_count


def test_dropping_a_source_drops_it_from_the_lockfile(corpus, workdir):
    ingest.ingest()
    assert "genai-platform" in lockfile()

    (workdir / "data" / "sources.yml").write_text(
        yaml.safe_dump({"sources": [corpus[0]]}), encoding="utf-8"
    )
    ingest.ingest()

    assert set(lockfile()) == {"field-guide"}


# --- failure modes --------------------------------------------------------

def test_an_unknown_source_type_is_not_silently_ignored(corpus, workdir):
    """A typo in `type:` currently falls through if/elif with no else: no
    fetch, no error, no lockfile entry. Silent data loss."""
    (workdir / "data" / "sources.yml").write_text(
        yaml.safe_dump({"sources": [{"id": "typo", "type": "gti", "url": "http://x"}]}),
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        fp.fetch_and_parse()


def test_the_raw_directory_setting_is_honoured_by_both_fetchers(
    workdir, monkeypatch, git_source, served_html
):
    """fetch_git writes to RAW, but fetch_url writes to a hardcoded
    "data/raw/...". They agree only while the setting keeps its default, and
    process_url reads from RAW -- so changing the setting breaks the pair."""
    monkeypatch.setattr(settings, "raw_data_dir", "data/custom_raw")
    module = importlib.reload(fp)
    custom = workdir / "data" / "custom_raw"
    try:
        module.fetch_git(git_source)
        assert list(custom.rglob("*.md")), "fetch_git ignored raw_data_dir"

        module.fetch_url({"id": "genai-platform", "type": "url", "url": served_html})
        assert (custom / "genai-platform.html").exists(), "fetch_url ignored raw_data_dir"
    finally:
        monkeypatch.undo()
        importlib.reload(fp)


# --- the corpus stays where the lockfile pins it --------------------------

def origin_of(source):
    from urllib.parse import urlparse
    from urllib.request import url2pathname
    return Path(url2pathname(urlparse(source["url"]).path))


def commit_change(repo, title):
    (repo / "guide" / "interview.md").write_text(long_markdown(title), encoding="utf-8")
    for args in (("add", "-A"), ("commit", "-qm", title)):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_a_new_upstream_commit_does_not_move_a_pinned_corpus(corpus, git_source):
    """Eval numbers only compare across runs if the corpus is identical."""
    fp.fetch_and_parse()
    pinned = lockfile()["field-guide"]["commit"]
    before = (PROCESSED / "field-guide" / "guide" / "interview.md").read_text(encoding="utf-8")

    commit_change(origin_of(git_source), "Rewritten Upstream")
    fp.fetch_and_parse()

    assert lockfile()["field-guide"]["commit"] == pinned
    assert (PROCESSED / "field-guide" / "guide" / "interview.md").read_text(encoding="utf-8") == before


def test_deleting_the_lockfile_lets_the_corpus_move_forward(corpus, git_source):
    """The pin is a floor, not a cage."""
    fp.fetch_and_parse()
    pinned = lockfile()["field-guide"]["commit"]

    commit_change(origin_of(git_source), "Rewritten Upstream")
    fp.LOCK.unlink()
    fp.fetch_and_parse()

    assert lockfile()["field-guide"]["commit"] != pinned
    assert "Rewritten Upstream" in (PROCESSED / "field-guide" / "guide" / "interview.md").read_text(encoding="utf-8")
