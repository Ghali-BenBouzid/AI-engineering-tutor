"""The lockfile is the record of which corpus version produced a result.

Its one job: a diff means the corpus actually moved. If fetched_at churned on
every run the file would be noise, and nobody would read the diff.
"""

import yaml

from src.fetch_and_parse import load_lock, lock_entry


def test_new_source_gets_a_timestamp():
    entry = lock_entry({}, {"url": "u", "sha256": "abc"})
    assert entry["fetched_at"]
    assert entry["sha256"] == "abc"


def test_unchanged_content_keeps_the_old_timestamp():
    first = lock_entry({}, {"url": "u", "sha256": "abc"})
    second = lock_entry(first, {"url": "u", "sha256": "abc"})
    assert second["fetched_at"] == first["fetched_at"]


def test_changed_content_stamps_a_new_timestamp():
    first = lock_entry({}, {"url": "u", "sha256": "abc"})
    first["fetched_at"] = "2000-01-01T00:00:00+00:00"
    second = lock_entry(first, {"url": "u", "sha256": "CHANGED"})
    assert second["fetched_at"] != first["fetched_at"]


def test_git_sources_are_versioned_by_commit():
    first = lock_entry({}, {"url": "u", "commit": "sha1", "files": 3})
    first["fetched_at"] = "2000-01-01T00:00:00+00:00"  # timestamps are second-resolution

    same = lock_entry(first, {"url": "u", "commit": "sha1", "files": 3})
    moved = lock_entry(first, {"url": "u", "commit": "sha2", "files": 3})

    assert same["fetched_at"] == first["fetched_at"]
    assert moved["fetched_at"] != first["fetched_at"]


def test_serialised_lockfile_is_byte_stable_across_runs():
    """Two runs over unchanged sources must produce an identical file,
    regardless of the order sources come back in."""
    run_one = {
        "b-source": lock_entry({}, {"url": "b", "sha256": "222"}),
        "a-source": lock_entry({}, {"url": "a", "commit": "111", "files": 2}),
    }
    dumped_one = yaml.safe_dump(run_one, sort_keys=True)

    run_two = {
        "a-source": lock_entry(run_one["a-source"],
                               {"url": "a", "commit": "111", "files": 2}),
        "b-source": lock_entry(run_one["b-source"], {"url": "b", "sha256": "222"}),
    }
    dumped_two = yaml.safe_dump(run_two, sort_keys=True)

    assert dumped_one == dumped_two


def test_load_lock_returns_empty_when_absent(workdir):
    assert load_lock() == {}


def test_load_lock_reads_what_was_written(workdir):
    entry = lock_entry({}, {"url": "u", "sha256": "abc"})
    (workdir / "data" / "sources.lock.yml").write_text(
        yaml.safe_dump({"src": entry}, sort_keys=True), encoding="utf-8"
    )
    assert load_lock()["src"]["sha256"] == "abc"
