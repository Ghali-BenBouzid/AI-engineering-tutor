"""Loads the golden set and its judgment pool.

Two files, because one is authored and the other is judged:
`golden.yml` holds the questions, `pool.yml` holds the ground truth.
"""

from pathlib import Path
from typing import Any

import yaml

GOLDEN = Path("evals/golden.yml")
POOL = Path("evals/pool.yml")

TYPES = {"conceptual", "document", "out_of_corpus"}
REQUIRED = {"id", "type", "question"}


def load_golden(path: Path = GOLDEN) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []

    seen: set[str] = set()
    for i, q in enumerate(questions):
        missing = REQUIRED - q.keys()
        if missing:
            raise ValueError(f"question {i}: missing {sorted(missing)}")
        if q["type"] not in TYPES:
            raise ValueError(f"{q['id']}: unknown type {q['type']!r}, expected one of {sorted(TYPES)}")
        if q["id"] in seen:
            raise ValueError(f"duplicate question id {q['id']!r}")
        seen.add(q["id"])

    return questions


def in_corpus(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [q for q in questions if q["type"] != "out_of_corpus"]


def load_pool(path: Path = POOL) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_pool(pool: dict[str, list[dict[str, Any]]], path: Path = POOL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(pool, sort_keys=False, allow_unicode=True, width=10**6),
        encoding="utf-8",
    )


def labels_for(qid: str, pool: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Candidates judged relevant. Unjudged (`relevant: null`) counts as not relevant."""
    return [c for c in pool.get(qid, []) if c.get("relevant") is True]
