"""Tier 1: the deterministic half of the evaluation.

No LLM, no judge, no API key. Retrieval either put a labelled section in the
top k or it did not, and the guardrail either fired or it did not. Both are
set arithmetic over a frozen corpus, so the numbers are identical run to run
and cheap enough to gate every commit.

Results are written to a committed snapshot. A commit that changes retrieval
changes that file, so the effect shows up as a reviewable diff - `q07 rank 3
-> rank 1` - instead of a metric moving for reasons nobody can see.

    uv run python -m evals.tier1            # compare against the snapshot
    uv run python -m evals.tier1 --update   # accept the current numbers
"""

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from evals.dataset import labels_for, load_golden, load_pool
from src.config import settings
from src.retrieval import should_abstain, top_k_search

SNAPSHOT = Path("evals/tier1_snapshot.json")

# Coarse absolute floors, set below the measured baseline. They catch a
# collapse - a broken index, a swapped model, an empty corpus - that a
# snapshot diff would also show but that nobody should be able to accept by
# rerunning with --update.
FLOORS = {"hit_rate_at_5": 0.60, "mrr": 0.45}


def mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def rank_of_first_hit(hits: list[tuple[Document, float]], labels: list[dict]) -> int | None:
    """Rank of the first retrieved chunk belonging to a labelled section.

    Section is matched as a substring so a label survives someone adding a
    heading level above it, the same reason labels name (doc_id, section)
    rather than a chunk id.
    """
    for rank, (doc, _) in enumerate(hits, start=1):
        section = doc.metadata.get("section", "")
        if any(doc.metadata["doc_id"] == label["doc_id"] and label["section"] in section
               for label in labels):
            return rank
    return None


def evaluate() -> dict[str, Any]:
    pool = load_pool()
    per_question, ranks, abstentions = {}, [], []

    for q in load_golden():
        hits = top_k_search(q["question"], top_k=settings.top_k)

        if q["type"] == "out_of_corpus":
            abstained = should_abstain(hits)
            per_question[q["id"]] = {"abstained": abstained}
            abstentions.append(abstained)
            continue

        labels = labels_for(q["id"], pool)
        assert labels, f"{q['id']} is {q['type']} but has no label"
        assert all(label["section"] for label in labels), f"{q['id']} has an empty section label"

        rank = rank_of_first_hit(hits, labels)
        per_question[q["id"]] = {"rank": rank, "hit": rank is not None}
        ranks.append(rank)

    return {
        # Recorded so a settings change lands in the same diff as the metric
        # movement it caused, rather than arriving unexplained.
        "config": {
            "top_k": settings.top_k,
            "max_distance": settings.max_distance,
            "embedding_model": settings.embedding_model_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "metrics": {
            f"hit_rate_at_{settings.top_k}": mean(rank is not None for rank in ranks),
            "mrr": mean(1 / rank if rank else 0.0 for rank in ranks),
            "abstention_rate": mean(abstentions),
        },
        "questions": per_question,
    }


def save(snapshot: dict[str, Any]) -> None:
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def differences(current: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    out = []
    for section in ("config", "metrics"):
        for key in sorted(current[section].keys() | previous[section].keys()):
            before, now = previous[section].get(key), current[section].get(key)
            if before != now:
                out.append(f"{section}.{key}: {before} -> {now}")

    for qid in sorted(current["questions"].keys() | previous["questions"].keys()):
        before, now = previous["questions"].get(qid), current["questions"].get(qid)
        if before != now:
            out.append(f"{qid}: {before} -> {now}")
    return out


def report(snapshot: dict[str, Any]) -> None:
    for key, value in snapshot["metrics"].items():
        print(f"{key:>18} = {value}")
    misses = [q for q, r in snapshot["questions"].items() if r.get("hit") is False]
    answered = [q for q, r in snapshot["questions"].items() if r.get("abstained") is False]
    print(f"\nretrieval missed: {', '.join(misses) or 'none'}")
    print(f"should have refused but answered: {', '.join(answered) or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write the snapshot instead of checking it")
    args = parser.parse_args()

    snapshot = evaluate()
    report(snapshot)

    if args.update:
        save(snapshot)
        print(f"\nwrote {SNAPSHOT}")
        return

    failures = [f"{name} = {snapshot['metrics'][name]} is below the floor of {floor}"
                for name, floor in FLOORS.items() if snapshot["metrics"][name] < floor]

    if not SNAPSHOT.exists():
        raise SystemExit(f"\n{SNAPSHOT} does not exist yet; run with --update")

    changes = differences(snapshot, json.loads(SNAPSHOT.read_text(encoding="utf-8")))
    if changes:
        print("\nsnapshot differs:")
        for line in changes:
            print(f"  {line}")
        failures.append("snapshot is out of date; review the diff, then rerun with --update")

    if failures:
        raise SystemExit("\n" + "\n".join(failures))
    print("\nsnapshot matches")


if __name__ == "__main__":
    main()
