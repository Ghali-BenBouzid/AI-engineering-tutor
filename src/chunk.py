import random
from pathlib import Path
from typing import Any

import frontmatter
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer

from src.config import settings

PROCESSED = Path(settings.processed_data_dir)
MODEL = settings.embedding_model_name
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CARRIED = ("author", "date")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS,
    strip_headers=False,
)

size_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=450,
    chunk_overlap=90,
)


def token_len(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_docs():
    for path in sorted(PROCESSED.rglob("*.md")):
        post = frontmatter.load(path)
        yield post.metadata, post.content


def chunk_doc(meta: dict[str, Any], body: str) -> list[dict[str, Any]]:
    records = []
    for section in header_splitter.split_text(body):
        path = " > ".join(
            section.metadata[k] for k in ("h1", "h2", "h3") if k in section.metadata
        )
        for text in size_splitter.split_text(section.page_content):
            records.append(
                {
                    "text": text,
                    "section": path,
                    "doc_id": meta["doc_id"],
                    "title": meta["title"],
                    "url": meta["url"],
                    "doc_hash": meta["doc_hash"],
                    **{k: meta[k] for k in CARRIED if meta.get(k) is not None},
                }
            )

    for i, record in enumerate(records):
        record["id"] = f"{record['doc_id']}:{i}"
    return records


def chunk_all() -> list[dict[str, Any]]:
    chunks = []
    for meta, body in load_docs():
        chunks += chunk_doc(meta, body)
    return chunks


def main() -> None:
    chunks = chunk_all()
    lengths = sorted(token_len(c["text"]) for c in chunks)

    oversized = [c for c in chunks if token_len(c["text"]) > tokenizer.model_max_length]
    assert not oversized, f"{len(oversized)} chunks exceed {tokenizer.model_max_length} tokens"

    print(f"{len(chunks)} chunks from {len(set(c['doc_id'] for c in chunks))} docs")
    print(f"tokens: min={lengths[0]} median={lengths[len(lengths) // 2]} max={lengths[-1]}")
    print(f"chunks under 30 tokens: {sum(1 for n in lengths if n < 30)}")

    print("\n--- 3 random chunks ---")
    for c in random.sample(chunks, 3):
        print(f"\n[{c['id']}] ({token_len(c['text'])} tok)  {c['section']}")
        print(c["text"])

    print("\n--- 5 shortest chunks ---")
    for c in sorted(chunks, key=lambda c: token_len(c["text"]))[:5]:
        print(f"[{token_len(c['text']):3d} tok] {c['id']} :: {c['text'][:90]!r}")


if __name__ == "__main__":
    main()
