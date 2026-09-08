"""Settings must load on a machine that has no .env -- CI, or a fresh clone."""

import pytest

from src.config import Settings


@pytest.fixture
def no_dotenv(workdir, monkeypatch):
    """A tmp cwd has no .env; clear the real environment too."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name, raising=False)
    return workdir


def test_settings_load_without_a_dotenv(no_dotenv):
    """A required secret would break `chunk.py`, which needs no API key."""
    settings = Settings()
    assert settings.openrouter_api_key is None


def test_secrets_are_optional_and_everything_else_has_a_default(no_dotenv):
    for name, field in Settings.model_fields.items():
        assert not field.is_required(), f"{name} is required; CI has no .env"


def test_environment_overrides_defaults(no_dotenv, monkeypatch):
    monkeypatch.setenv("TOP_K", "42")
    assert Settings().top_k == 42


def test_values_are_coerced_to_their_declared_types(no_dotenv, monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "300")
    settings = Settings()
    assert settings.chunk_size == 300
    assert isinstance(settings.chunk_size, int)


def test_overlap_is_smaller_than_chunk_size(no_dotenv):
    settings = Settings()
    assert 0 <= settings.chunk_overlap < settings.chunk_size
