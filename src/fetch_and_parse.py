from trafilatura import extract
import httpx
import yaml
import shutil
import subprocess
from fnmatch import fnmatch
from pathlib import Path
import re
import frontmatter
import hashlib

from typing import Any
from datetime import datetime, timezone


CACHE = Path(".cache")
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
LOCK = Path("data/sources.lock.yml")


def load_lock() -> dict[str, str]:
    if not LOCK.exists():
        return {}
    return yaml.safe_load(LOCK.read_text(encoding="utf-8")) or {}


def lock_entry(previous, fields) -> dict[str, str]:
    """Only stamp a new fetched_at when the content actually changed,
    so a lockfile diff always means the corpus moved."""
    version = fields.get("sha256") or fields.get("commit")
    prev_version = previous.get("sha256") or previous.get("commit")

    if previous and version == prev_version:
        fields["fetched_at"] = previous["fetched_at"]
    else:
        fields["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return fields

def fetch_git(source: dict[str, Any]) -> str:
    dest = CACHE / source["id"]

    if not dest.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", source["url"], str(dest)],
            check=True,
        )

    sha = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    excludes = source.get("exclude", [])
    for md in sorted(dest.rglob("*.md")):
        rel = md.relative_to(dest)
        if any(fnmatch(str(rel), pat) for pat in excludes):
            continue
        out = RAW / source["id"] / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(md, out)

    return sha


def first_heading(text) -> str | None:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def process_git(source, sha) -> None:
    root = RAW / source["id"]
    for md in sorted(root.rglob("*.md")):
        rel = md.relative_to(root)
        post = frontmatter.loads(md.read_text(encoding="utf-8"))
        body = post.content                       # frontmatter already stripped
        doc_id = f"{source['id']}/{rel.with_suffix('')}"

        meta = {
            "doc_id": doc_id,
            "title": post.get("title") or first_heading(body) or rel.stem,
            "url": f"{source['url']}/blob/{sha}/{rel.as_posix()}",
            "doc_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }

        out = PROCESSED / f"{doc_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(frontmatter.dumps(frontmatter.Post(body, **meta)),
                       encoding="utf-8")


def fetch_url(source: dict[str, Any]) -> str:
    downloaded = httpx.get(source["url"], follow_redirects=True, timeout=30.0).text

    with open(f"data/raw/{source["id"]}.html", 'w') as f:
        f.write(downloaded)

    return hashlib.sha256(downloaded.encode("utf-8")).hexdigest()


def process_url(source: dict[str, Any]) -> None:
    html = (RAW / f"{source["id"]}.html").read_text(encoding="utf-8")
    markdown = extract(html, with_metadata=True, output_format="markdown")

    post = frontmatter.loads(markdown)
    body = post.content

    meta = dict(post.metadata)                    # keep author, date, ...
    meta.update({
        "doc_id": source["id"],
        "title": meta.get("title") or first_heading(body) or source["id"],
        "url": source["url"],
        "doc_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    })

    out = PROCESSED / f"{source["id"]}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(frontmatter.dumps(frontmatter.Post(body, **meta)),
                   encoding="utf-8")


def fetch_and_parse() -> None:
    with open("data/sources.yml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

        sources_list = data_loaded["sources"]

        previous_lock = load_lock()
        lock = {}

        for source in sources_list:
            if source["type"] == "url":
                sha256 = fetch_url(source)
                process_url(source)

                lock[source["id"]] = lock_entry(
                    previous_lock.get(source["id"], {}),
                    {
                        "url": source["url"],
                        "sha256": sha256,
                    },
                )

            elif source["type"] == "git":
                sha = fetch_git(source)
                process_git(source, sha)

                lock[source["id"]] = lock_entry(
                    previous_lock.get(source["id"], {}),
                    {
                        "url": source["url"],
                        "commit": sha,
                        "files": len(list((RAW / source["id"]).rglob("*.md"))),
                    },
                )

        LOCK.write_text(yaml.safe_dump(lock, sort_keys=True), encoding="utf-8")


if __name__=="__main__":
    fetch_and_parse()