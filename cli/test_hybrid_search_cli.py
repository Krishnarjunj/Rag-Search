import importlib
import os
import sys
import types
from pathlib import Path


CLI_DIR = Path(__file__).resolve().parent
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))


def load_cli_module():
    fake_hybrid_search = types.ModuleType("hybrid_search")

    class PlaceholderHybridSearch:
        def __init__(self, documents):
            self.documents = documents

        def rrf_search(self, query, k, limit):
            return []

    fake_hybrid_search.HybridSearch = PlaceholderHybridSearch
    sys.modules["hybrid_search"] = fake_hybrid_search
    sys.modules.pop("hybrid_search_cli", None)
    return importlib.import_module("hybrid_search_cli")


def test_rrf_search_without_enhancement(monkeypatch, capsys):
    hybrid_search_cli = load_cli_module()

    search_calls = []

    class FakeHybridSearch:
        def __init__(self, documents):
            self.documents = documents

        def rrf_search(self, query, k, limit):
            search_calls.append((query, k, limit))
            return [
                {
                    "title": "Inception",
                    "rrf_score": 0.91,
                    "bm25_rank": 1,
                    "semantic_rank": 2,
                    "document": "A dream heist thriller.",
                }
            ]

    monkeypatch.setattr(hybrid_search_cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(hybrid_search_cli, "load_movies", lambda: [{"title": "Inception"}])

    monkeypatch.setattr(sys, "argv", ["hybrid_search_cli.py", "rrf-search", "inceoption"])
    hybrid_search_cli.main()

    output = capsys.readouterr().out
    assert "Enhanced query" not in output
    assert "1. Inception" in output
    assert search_calls == [("inceoption", 60, 5)]


def test_rrf_search_with_spell_enhancement(monkeypatch, capsys):
    hybrid_search_cli = load_cli_module()

    search_calls = []
    enhance_calls = []

    class FakeHybridSearch:
        def __init__(self, documents):
            self.documents = documents

        def rrf_search(self, query, k, limit):
            search_calls.append((query, k, limit))
            return [
                {
                    "title": "Interstellar",
                    "rrf_score": 0.88,
                    "bm25_rank": 2,
                    "semantic_rank": 1,
                    "document": "Explorers travel through a wormhole.",
                }
            ]

    def fake_enhance_query(query, method):
        enhance_calls.append((query, method))
        return "interstellar"

    monkeypatch.setattr(hybrid_search_cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(hybrid_search_cli, "enhance_query", fake_enhance_query)
    monkeypatch.setattr(hybrid_search_cli, "load_movies", lambda: [{"title": "Interstellar"}])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_search_cli.py",
            "rrf-search",
            "interstller",
            "--enhance",
            "spell",
            "--limit",
            "3",
        ],
    )
    hybrid_search_cli.main()

    output = capsys.readouterr().out
    assert "Enhanced query (spell): 'interstller' -> 'interstellar'" in output
    assert "1. Interstellar" in output
    assert enhance_calls == [("interstller", "spell")]
    assert search_calls == [("interstellar", 60, 3)]


def test_rrf_search_with_spell_enhancement_without_api_key(monkeypatch, capsys):
    hybrid_search_cli = load_cli_module()

    search_calls = []

    class FakeHybridSearch:
        def __init__(self, documents):
            self.documents = documents

        def rrf_search(self, query, k, limit):
            search_calls.append((query, k, limit))
            return [
                {
                    "title": "Paddington",
                    "rrf_score": 0.77,
                    "bm25_rank": 1,
                    "semantic_rank": 1,
                    "document": "A British bear finds a home in London.",
                }
            ]

    monkeypatch.setattr(hybrid_search_cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(hybrid_search_cli, "load_movies", lambda: [{"title": "Paddington"}])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_search_cli.py",
            "rrf-search",
            "briish bear",
            "--enhance",
            "spell",
        ],
    )
    hybrid_search_cli.main()

    output = capsys.readouterr().out
    assert "Enhanced query (spell): 'briish bear' -> 'briish bear'" in output
    assert "1. Paddington" in output
    assert search_calls == [("briish bear", 60, 5)]
