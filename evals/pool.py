"""Builds the judgment pool for the golden set.

Labelling every relevant chunk in the corpus is infeasible, so we pool: take
the top candidates from more than one retrieval method, judge only those, and
treat everything unjudged as not relevant. Unjudged-means-irrelevant can only
understate a score, never inflate one, which is the safe direction for a gate.

Two methods rather than one, because a pool built by a single retriever
flatters that retriever: anything only a different method would have found is
never labelled, so the different method scores as a miss. Dense catches
paraphrase, BM25 catches exact terms, and `found_by` records which - that field
is the evidence for or against hybrid search in V1.

Pooling only ever adds. A candidate you have judged is kept forever, even once
it drops out of the pool, so the labels get more complete with every retrieval
change instead of freezing around whatever today's retriever likes.

    uv run python -m evals.pool
"""

import re
from typing import Any

from rank_bm25 import BM25Okapi

from evals.dataset import load_golden, load_pool, save_pool
from src.chunk import chunk_all
from src.retrieval import top_k_search

POOL_SIZE = 20
RRF_K = 60      # reciprocal rank fusion damping; 60 is the value from the original paper
PREVIEW_CHARS = 240

WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


def preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:PREVIEW_CHARS] + ("..." if len(flat) > PREVIEW_CHARS else "")


def bm25_top(bm25: BM25Okapi, chunks: list[dict[str, Any]], question: str, n: int):
    scores = bm25.get_scores(tokenize(question))
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:n]
    return [(chunks[i]["doc_id"], chunks[i]["section"], chunks[i]["text"]) for i in order]


def dense_top(question: str, n: int):
    return [
        (doc.metadata["doc_id"], doc.metadata.get("section", ""), doc.page_content)
        for doc, _ in top_k_search(question, top_k=n)
    ]


def fuse(rankings: dict[str, list[tuple[str, str, str]]]) -> list[dict[str, Any]]:
    """Merge per-method hits into one section-level candidate list, best first.

    Candidates are sections, not chunks, because labels are (doc_id, section):
    a label that named a chunk id would die at the next rechunk. Several chunks
    from one section therefore collapse to a single candidate, and only the
    first occurrence takes up a rank slot.

    Ordering is reciprocal rank fusion, so a section both methods found floats
    above one either found alone. It only decides reading order - what you tick
    is what counts.
    """
    fused: dict[tuple[str, str], dict[str, Any]] = {}

    for method, hits in rankings.items():
        rank = 0
        for doc_id, section, text in hits:
            key = (doc_id, section)
            entry = fused.get(key)
            if entry is not None and method in entry["found_by"]:
                continue
            rank += 1
            if entry is None:
                entry = fused[key] = {
                    "doc_id": doc_id,
                    "section": section,
                    "found_by": [],
                    "preview": preview(text),
                    "_score": 0.0,
                }
            entry["found_by"].append(method)
            entry["_score"] += 1 / (RRF_K + rank)

    ordered = sorted(fused.values(), key=lambda e: -e["_score"])
    for entry in ordered:
        del entry["_score"]
    return ordered


def merge(judged: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Carry existing judgments onto a freshly built pool.

    A judged candidate survives even if this pool no longer contains it, so a
    retrieval change can add ground truth but never silently delete it.
    Unjudged candidates that fell out carry no information and are dropped.
    """
    previous = {(c["doc_id"], c["section"]): c for c in judged}

    merged = []
    for candidate in fresh:
        old = previous.pop((candidate["doc_id"], candidate["section"]), None)
        candidate["relevant"] = old.get("relevant") if old else None
        merged.append(candidate)

    merged.extend(c for c in previous.values() if c.get("relevant") is not None)
    return merged


def build() -> dict[str, list[dict[str, Any]]]:
    questions = load_golden()
    if not questions:
        raise SystemExit("evals/golden.yml has no questions yet")

    chunks = chunk_all()
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    existing = load_pool()

    pool = {}
    for q in questions:
        if q["type"] == "out_of_corpus":
            continue        # nothing to label: the assertion is that we refuse
        fresh = fuse({
            "dense": dense_top(q["question"], POOL_SIZE),
            "bm25": bm25_top(bm25, chunks, q["question"], POOL_SIZE),
        })
        pool[q["id"]] = merge(existing.get(q["id"], []), fresh)
    return pool


def main() -> None:
    pool = build()
    save_pool(pool)

    unjudged = 0
    for qid, candidates in pool.items():
        marked = sum(1 for c in candidates if c.get("relevant") is True)
        blank = sum(1 for c in candidates if c.get("relevant") is None)
        unjudged += blank
        print(f"{qid}  {len(candidates):3d} candidates  {marked} relevant  {blank} unjudged")

    print(f"\n{len(pool)} questions pooled, {unjudged} candidates awaiting judgment")
    print("Mark the ones that answer the question with `relevant: true` in evals/pool.yml.")


if __name__ == "__main__":
    main()
