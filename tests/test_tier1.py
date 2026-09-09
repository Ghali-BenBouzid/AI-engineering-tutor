"""Tier 1 scoring, exercised without an index or a model.

The scorer decides what every future retrieval change is measured against, so
it gets tested on its own rather than only through a full run.
"""

import json

import pytest
from langchain_core.documents import Document

from evals.dataset import load_golden
from evals.tier1 import FLOORS, SNAPSHOT, differences, mean, rank_of_first_hit
from src.config import settings


def hit(doc_id, section):
    return (Document(page_content="", metadata={"doc_id": doc_id, "section": section}), 0.1)


LABEL = [{"doc_id": "d", "section": "Guide > Chunking"}]


def test_rank_is_one_based():
    assert rank_of_first_hit([hit("d", "Guide > Chunking")], LABEL) == 1


def test_rank_counts_past_unlabelled_hits():
    hits = [hit("d", "Guide > Other"), hit("x", "Guide > Chunking"), hit("d", "Guide > Chunking")]
    assert rank_of_first_hit(hits, LABEL) == 3


def test_a_miss_is_none_rather_than_zero():
    """None and 0 would both be falsy, but only one of them is a rank."""
    assert rank_of_first_hit([hit("d", "Guide > Other")], LABEL) is None


def test_the_same_section_in_another_document_is_not_a_hit():
    assert rank_of_first_hit([hit("other", "Guide > Chunking")], LABEL) is None


def test_a_label_survives_a_heading_level_being_added_above_it():
    """Why labels are matched as substrings and never as chunk ids."""
    assert rank_of_first_hit([hit("d", "Manual > Guide > Chunking")], LABEL) == 1


def test_substring_matching_also_matches_a_longer_sibling_name():
    """The accepted cost of substring labels: a sibling whose name extends the
    label counts as a hit. Tolerated because labels come from observed section
    paths, so a collision needs a real heading that starts with another one."""
    assert rank_of_first_hit([hit("d", "Guide > Chunking Strategies")], LABEL) == 1
    assert rank_of_first_hit([hit("d", "Guide > Chunk")], LABEL) is None


def test_any_one_label_is_enough():
    labels = [{"doc_id": "d", "section": "A"}, {"doc_id": "d", "section": "B"}]
    assert rank_of_first_hit([hit("d", "B")], labels) == 1


def test_mean_of_nothing_is_zero_rather_than_a_crash():
    assert mean([]) == 0.0


def test_differences_reports_a_moved_question():
    before = {"config": {}, "metrics": {}, "questions": {"q1": {"rank": 3, "hit": True}}}
    after = {"config": {}, "metrics": {}, "questions": {"q1": {"rank": 1, "hit": True}}}

    assert differences(after, before) == ["q1: {'rank': 3, 'hit': True} -> {'rank': 1, 'hit': True}"]


def test_differences_is_empty_when_nothing_moved():
    same = {"config": {"top_k": 5}, "metrics": {"mrr": 0.5}, "questions": {"q1": {"rank": 1}}}
    assert differences(same, same) == []


def test_the_committed_snapshot_matches_the_current_settings():
    """Catches a settings change committed without rerunning the eval."""
    config = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["config"]

    assert config["top_k"] == settings.top_k
    assert config["chunk_size"] == settings.chunk_size
    assert config["chunk_overlap"] == settings.chunk_overlap
    assert config["embedding_model"] == settings.embedding_model_name
    assert config["max_distance"] == settings.max_distance


def test_the_snapshot_covers_every_question():
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert snapshot["questions"].keys() == {q["id"] for q in load_golden()}


@pytest.mark.parametrize("name", FLOORS)
def test_every_floor_names_a_metric_that_exists(name):
    """A floor on a misspelled metric silently guards nothing."""
    metrics = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["metrics"]
    assert name in metrics, f"floor {name!r} matches no metric in {sorted(metrics)}"
    assert metrics[name] >= FLOORS[name]
