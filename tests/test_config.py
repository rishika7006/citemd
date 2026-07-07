import importlib

from citemd.config import Settings


def test_defaults():
    # Ignore any local .env so this asserts the code-defined defaults deterministically.
    s = Settings(_env_file=None)
    assert s.embedding_dim == 384
    assert s.llm_model == "claude-sonnet-4-6"
    assert s.rrf_k == 60
    assert "citemd" in s.database_url


def test_env_override(monkeypatch):
    monkeypatch.setenv("CITEMD_LLM_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("CITEMD_RETRIEVE_K", "7")
    s = Settings()
    assert s.llm_model == "claude-haiku-4-5"
    assert s.retrieve_k == 7


def test_version_exposed():
    mod = importlib.import_module("citemd")
    assert mod.__version__ == "0.1.0"
