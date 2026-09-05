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


CACHE = Path(".cache")
RAW = Path("data/raw")
PROCESSED = Path("data/processed")

def fetch_git(source: dict[str, Any]):
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


def first_heading(text):
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def process_git(source, sha):
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


def main() -> None:
    with open("data/sources.yml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

        sources_list = data_loaded["sources"]

        for source in sources_list:
            if source["type"] == "url":
                downloaded = httpx.get(source["url"], follow_redirects=True, timeout=30.0).text

                with open(f"data/raw/{source["id"]}.html", 'w') as f:
                    f.write(downloaded)

                    result = extract(downloaded, with_metadata=True, output_format="markdown")
                    with open(f"data/processed/{source["id"]}.md", 'w') as f:
                        f.write(result)

            elif source["type"] == "git":
                sha = fetch_git(source)
                process_git(source, sha)


if __name__=="__main__":
    main()