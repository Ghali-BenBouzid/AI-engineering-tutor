"""The pooling logic, which decides what ground truth exists at all.

Pure functions only: no index, no model, no network. The two properties that
matter are that candidates are sections rather than chunks, and that a
judgment is never lost when the pool is rebuilt.
"""

import pytest
import yaml

from evals.dataset import in_corpus, labels_for, load_golden, load_pool, save_pool
from evals.pool import fuse, merge


def hit(doc_id, section, text="some text"):
    return (doc_id, section, text)


def test_several_chunks_of_one_section_collapse_to_one_candidate():
    """Labels are (doc_id, section), so a section is the unit you judge."""
    candidates = fuse({"dense": [hit("d", "A", "first chunk"), hit("d", "A", "second chunk")]})

    assert len(candidates) == 1
    assert (candidates[0]["doc_id"], candidates[0]["section"]) == ("d", "A")


def test_same_section_in_two_documents_stays_two_candidates():
    candidates = fuse({"dense": [hit("d1", "Overview"), hit("d2", "Overview")]})
    assert len(candidates) == 2


def test_found_by_records_every_method_that_surfaced_it():
    candidates = fuse({"dense": [hit("d", "A")], "bm25": [hit("d", "A")]})

    assert candidates[0]["found_by"] == ["dense", "bm25"]


def test_a_section_both_methods_found_is_read_first():
    candidates = fuse({
        "dense": [hit("d", "only-dense"), hit("d", "both")],
        "bm25": [hit("d", "both"), hit("d", "only-bm25")],
    })

    assert candidates[0]["section"] == "both"


def test_a_duplicate_within_one_method_does_not_consume_a_rank():
    """Otherwise a section repeated by one method would push down everything after it."""
    with_duplicate = fuse({"dense": [hit("d", "A"), hit("d", "A"), hit("d", "B")]})
    without = fuse({"dense": [hit("d", "A"), hit("d", "B")]})

    assert [c["section"] for c in with_duplicate] == [c["section"] for c in without]


def test_a_new_candidate_starts_unjudged():
    assert merge([], fuse({"dense": [hit("d", "A")]}))[0]["relevant"] is None


def test_rebuilding_the_pool_keeps_an_existing_judgment():
    judged = merge([], fuse({"dense": [hit("d", "A")]}))
    judged[0]["relevant"] = True

    rebuilt = merge(judged, fuse({"dense": [hit("d", "A")]}))

    assert rebuilt[0]["relevant"] is True


def test_a_judged_candidate_survives_falling_out_of_the_pool():
    """Ground truth outlives the retriever that happened to surface it."""
    judged = [{"doc_id": "d", "section": "A", "found_by": ["bm25"], "relevant": True}]

    rebuilt = merge(judged, fuse({"dense": [hit("d", "B")]}))

    assert {(c["doc_id"], c["section"]): c["relevant"] for c in rebuilt} == {
        ("d", "A"): True,
        ("d", "B"): None,
    }


def test_an_unjudged_candidate_that_fell_out_is_dropped():
    stale = [{"doc_id": "d", "section": "A", "found_by": ["dense"], "relevant": None}]

    rebuilt = merge(stale, fuse({"dense": [hit("d", "B")]}))

    assert [c["section"] for c in rebuilt] == ["B"]


def test_only_true_counts_as_a_label():
    pool = {"q1": [
        {"doc_id": "d", "section": "yes", "relevant": True},
        {"doc_id": "d", "section": "no", "relevant": False},
        {"doc_id": "d", "section": "unjudged", "relevant": None},
    ]}

    assert [c["section"] for c in labels_for("q1", pool)] == ["yes"]


def test_a_pool_survives_a_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "pool.yml"
    pool = {"q1": merge([], fuse({"dense": [hit("d", "A", "a preview with  odd\nspacing")]}))}

    save_pool(pool, path)

    assert load_pool(path) == pool


def test_an_absent_pool_is_empty_rather_than_an_error(tmp_path):
    assert load_pool(tmp_path / "nothing.yml") == {}


def test_the_committed_golden_set_is_valid():
    """Runs against the real file, so a bad question fails the suite."""
    load_golden()


def golden(tmp_path, questions):
    path = tmp_path / "golden.yml"
    path.write_text(yaml.safe_dump({"questions": questions}), encoding="utf-8")
    return path


def test_duplicate_question_ids_are_rejected(tmp_path):
    path = golden(tmp_path, [
        {"id": "q1", "type": "conceptual", "question": "a"},
        {"id": "q1", "type": "conceptual", "question": "b"},
    ])

    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)


def test_an_unknown_question_type_is_rejected(tmp_path):
    path = golden(tmp_path, [{"id": "q1", "type": "tricky", "question": "a"}])

    with pytest.raises(ValueError, match="unknown type"):
        load_golden(path)


def test_a_question_missing_a_field_is_rejected(tmp_path):
    path = golden(tmp_path, [{"id": "q1", "type": "conceptual"}])

    with pytest.raises(ValueError, match="question"):
        load_golden(path)


def test_every_in_corpus_question_has_at_least_one_label():
    """A question with no label is not answerable, so it belongs in out_of_corpus."""
    pool = load_pool()
    for q in in_corpus(load_golden()):
        assert labels_for(q["id"], pool), f"{q['id']} has no label but is typed {q['type']!r}"


def test_out_of_corpus_questions_carry_no_labels():
    pool = load_pool()
    for q in load_golden():
        if q["type"] == "out_of_corpus":
            assert not pool.get(q["id"]), f"{q['id']} is out_of_corpus but has candidates"


@pytest.mark.slow
def test_every_label_points_at_a_section_that_still_exists():
    """Labels are (doc_id, section) so they survive rechunking - this proves it."""
    from src.chunk import chunk_all

    sections = {(c["doc_id"], c["section"]) for c in chunk_all()}
    pool = load_pool()
    for qid in pool:
        for label in labels_for(qid, pool):
            key = (label["doc_id"], label["section"])
            assert key in sections, f"{qid}: label {key} no longer exists in the corpus"
