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
    assert "Reciprocal Rank Fusion Results for 'inceoption' (k=60):" in output
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
    assert "Reciprocal Rank Fusion Results for 'interstellar' (k=60):" in output
    assert "1. Interstellar" in output
    assert enhance_calls == [("interstller", "spell")]
    assert search_calls == [("interstellar", 60, 3)]


def test_rrf_search_with_rewrite_enhancement(monkeypatch, capsys):
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
                    "title": "The Revenant",
                    "rrf_score": 0.93,
                    "bm25_rank": 1,
                    "semantic_rank": 1,
                    "document": "A frontiersman survives a brutal bear attack.",
                }
            ]

    def fake_enhance_query(query, method):
        enhance_calls.append((query, method))
        return "The Revenant Leonardo DiCaprio bear attack"

    monkeypatch.setattr(hybrid_search_cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(hybrid_search_cli, "enhance_query", fake_enhance_query)
    monkeypatch.setattr(hybrid_search_cli, "load_movies", lambda: [{"title": "The Revenant"}])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_search_cli.py",
            "rrf-search",
            "that bear movie where leo gets attacked",
            "--enhance",
            "rewrite",
        ],
    )
    hybrid_search_cli.main()

    output = capsys.readouterr().out
    assert (
        "Enhanced query (rewrite): 'that bear movie where leo gets attacked' -> "
        "'The Revenant Leonardo DiCaprio bear attack'" in output
    )
    assert "Reciprocal Rank Fusion Results for 'The Revenant Leonardo DiCaprio bear attack' (k=60):" in output
    assert "1. The Revenant" in output
    assert enhance_calls == [("that bear movie where leo gets attacked", "rewrite")]
    assert search_calls == [("The Revenant Leonardo DiCaprio bear attack", 60, 5)]


def test_rrf_search_with_expand_enhancement(monkeypatch, capsys):
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
                    "title": "Good Will Hunting",
                    "rrf_score": 0.89,
                    "bm25_rank": 3,
                    "semantic_rank": 1,
                    "document": "A gifted young janitor struggles with mathematics and identity.",
                }
            ]

    def fake_enhance_query(query, method):
        enhance_calls.append((query, method))
        return "math movie mathematics genius prodigy education"

    monkeypatch.setattr(hybrid_search_cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(hybrid_search_cli, "enhance_query", fake_enhance_query)
    monkeypatch.setattr(hybrid_search_cli, "load_movies", lambda: [{"title": "Good Will Hunting"}])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_search_cli.py",
            "rrf-search",
            "math movie",
            "--enhance",
            "expand",
            "--limit",
            "25",
        ],
    )
    hybrid_search_cli.main()

    output = capsys.readouterr().out
    assert (
        "Enhanced query (expand): 'math movie' -> "
        "'math movie mathematics genius prodigy education'" in output
    )
    assert "Reciprocal Rank Fusion Results for 'math movie mathematics genius prodigy education' (k=60):" in output
    assert "1. Good Will Hunting" in output
    assert enhance_calls == [("math movie", "expand")]
    assert search_calls == [("math movie mathematics genius prodigy education", 60, 25)]


def test_rrf_search_with_individual_reranking(monkeypatch, capsys):
    hybrid_search_cli = load_cli_module()

    search_calls = []
    rerank_calls = []

    class FakeHybridSearch:
        def __init__(self, documents):
            self.documents = documents

        def rrf_search(self, query, k, limit):
            search_calls.append((query, k, limit))
            return [
                {
                    "title": "The Country Bears",
                    "rrf_score": 0.023,
                    "bm25_rank": 25,
                    "semantic_rank": 32,
                    "document": "A young bear raised by a human family searches for belonging.",
                },
                {
                    "title": "The Berenstain Bears' Christmas Tree",
                    "rrf_score": 0.027,
                    "bm25_rank": 37,
                    "semantic_rank": 1,
                    "document": "It is Christmas Eve in Bear Country and the Bear Family is decorating.",
                },
            ]

    def fake_rerank_results(query, results, method):
        rerank_calls.append((query, method, len(results)))
        updated_results = []
        for result in results:
            updated_result = dict(result)
            updated_result["rerank_score"] = (
                10.0 if result["title"] == "The Berenstain Bears' Christmas Tree" else 9.0
            )
            updated_results.append(updated_result)
        updated_results.sort(key=lambda result: result["rerank_score"], reverse=True)
        return updated_results

    monkeypatch.setattr(hybrid_search_cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(hybrid_search_cli, "rerank_results", fake_rerank_results)
    monkeypatch.setattr(
        hybrid_search_cli, "load_movies", lambda: [{"title": "The Berenstain Bears' Christmas Tree"}]
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_search_cli.py",
            "rrf-search",
            "family movie about bears in the woods",
            "--rerank-method",
            "individual",
            "--limit",
            "3",
        ],
    )
    hybrid_search_cli.main()

    output = capsys.readouterr().out
    assert "Re-ranking top 3 results using individual method..." in output
    assert "Reciprocal Rank Fusion Results for 'family movie about bears in the woods' (k=60):" in output
    assert "1. The Berenstain Bears' Christmas Tree" in output
    assert "Re-rank Score: 10.000/10" in output
    assert search_calls == [("family movie about bears in the woods", 60, 15)]
    assert rerank_calls == [("family movie about bears in the woods", "individual", 2)]


def test_rrf_search_with_batch_reranking(monkeypatch, capsys):
    hybrid_search_cli = load_cli_module()

    search_calls = []
    rerank_calls = []

    class FakeHybridSearch:
        def __init__(self, documents):
            self.documents = documents

        def rrf_search(self, query, k, limit):
            search_calls.append((query, k, limit))
            return [
                {
                    "title": "The Country Bears",
                    "rrf_score": 0.023,
                    "bm25_rank": 25,
                    "semantic_rank": 32,
                    "document": "A young bear raised by a human family searches for belonging.",
                },
                {
                    "title": "The Berenstain Bears' Christmas Tree",
                    "rrf_score": 0.027,
                    "bm25_rank": 37,
                    "semantic_rank": 1,
                    "document": "It is Christmas Eve in Bear Country and the Bear Family is decorating.",
                },
                {
                    "title": "Goldilocks and the Three Bears",
                    "rrf_score": 0.023,
                    "bm25_rank": 2,
                    "semantic_rank": 91,
                    "document": "Three anthropomorphic bears encounter Goldilocks in the woods.",
                },
            ]

    def fake_rerank_results(query, results, method):
        rerank_calls.append((query, method, len(results)))
        reordered = []
        for rank, result in enumerate([results[1], results[2], results[0]], start=1):
            updated_result = dict(result)
            updated_result["rerank_rank"] = rank
            reordered.append(updated_result)
        return reordered

    monkeypatch.setattr(hybrid_search_cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(hybrid_search_cli, "rerank_results", fake_rerank_results)
    monkeypatch.setattr(
        hybrid_search_cli, "load_movies", lambda: [{"title": "The Berenstain Bears' Christmas Tree"}]
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_search_cli.py",
            "rrf-search",
            "family movie about bears in the woods",
            "--rerank-method",
            "batch",
            "--limit",
            "3",
        ],
    )
    hybrid_search_cli.main()

    output = capsys.readouterr().out
    assert "Re-ranking top 3 results using batch method..." in output
    assert "Reciprocal Rank Fusion Results for 'family movie about bears in the woods' (k=60):" in output
    assert "1. The Berenstain Bears' Christmas Tree" in output
    assert "Re-rank Rank: 1" in output
    assert search_calls == [("family movie about bears in the woods", 60, 15)]
    assert rerank_calls == [("family movie about bears in the woods", "batch", 3)]


def test_rrf_search_with_cross_encoder_reranking(monkeypatch, capsys):
    hybrid_search_cli = load_cli_module()

    search_calls = []
    rerank_calls = []

    class FakeHybridSearch:
        def __init__(self, documents):
            self.documents = documents

        def rrf_search(self, query, k, limit):
            search_calls.append((query, k, limit))
            return [
                {
                    "title": "The Country Bears",
                    "rrf_score": 0.023,
                    "bm25_rank": 25,
                    "semantic_rank": 32,
                    "document": "A young bear raised by a human family searches for belonging.",
                },
                {
                    "title": "Care Bears Movie II: A New Generation",
                    "rrf_score": 0.028,
                    "bm25_rank": 1,
                    "semantic_rank": 26,
                    "document": "A yellow bear and a purple horse look after baby animals.",
                },
                {
                    "title": "A Bear for Punishment",
                    "rrf_score": 0.020,
                    "bm25_rank": 41,
                    "semantic_rank": 35,
                    "document": "The bear family sleeps peacefully before alarms go off.",
                },
            ]

    def fake_rerank_results(query, results, method):
        rerank_calls.append((query, method, len(results)))
        reordered = []
        for score, result in [
            (3.314, results[1]),
            (-1.001, results[2]),
            (-1.276, results[0]),
        ]:
            updated_result = dict(result)
            updated_result["cross_encoder_score"] = score
            reordered.append(updated_result)
        return reordered

    monkeypatch.setattr(hybrid_search_cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(hybrid_search_cli, "rerank_results", fake_rerank_results)
    monkeypatch.setattr(
        hybrid_search_cli, "load_movies", lambda: [{"title": "Care Bears Movie II: A New Generation"}]
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_search_cli.py",
            "rrf-search",
            "family movie about bears in the woods",
            "--rerank-method",
            "cross_encoder",
            "--limit",
            "25",
        ],
    )
    hybrid_search_cli.main()

    output = capsys.readouterr().out
    assert "Re-ranking top 25 results using cross_encoder method..." in output
    assert "Reciprocal Rank Fusion Results for 'family movie about bears in the woods' (k=60):" in output
    assert "1. Care Bears Movie II: A New Generation" in output
    assert "Cross Encoder Score: 3.314" in output
    assert search_calls == [("family movie about bears in the woods", 60, 125)]
    assert rerank_calls == [("family movie about bears in the woods", "cross_encoder", 3)]


def test_rrf_search_with_evaluation(monkeypatch, capsys):
    hybrid_search_cli = load_cli_module()

    search_calls = []
    evaluation_calls = []

    class FakeHybridSearch:
        def __init__(self, documents):
            self.documents = documents

        def rrf_search(self, query, k, limit):
            search_calls.append((query, k, limit))
            return [
                {
                    "title": "The Berenstain Bears' Christmas Tree",
                    "rrf_score": 0.027,
                    "bm25_rank": 37,
                    "semantic_rank": 1,
                    "document": "It is Christmas Eve in Bear Country and the Bear Family is decorating.",
                },
                {
                    "title": "The Country Bears",
                    "rrf_score": 0.023,
                    "bm25_rank": 25,
                    "semantic_rank": 32,
                    "document": "A young bear raised by a human family searches for belonging.",
                },
            ]

    def fake_evaluate_results(query, results):
        evaluation_calls.append((query, len(results)))
        return [3, 2]

    monkeypatch.setattr(hybrid_search_cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(hybrid_search_cli, "evaluate_results", fake_evaluate_results)
    monkeypatch.setattr(
        hybrid_search_cli, "load_movies", lambda: [{"title": "The Berenstain Bears' Christmas Tree"}]
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_search_cli.py",
            "rrf-search",
            "family movie about bears in the woods",
            "--evaluate",
        ],
    )
    hybrid_search_cli.main()

    output = capsys.readouterr().out
    assert "Reciprocal Rank Fusion Results for 'family movie about bears in the woods' (k=60):" in output
    assert "1. The Berenstain Bears' Christmas Tree: 3/3" in output
    assert "2. The Country Bears: 2/3" in output
    assert search_calls == [("family movie about bears in the woods", 60, 5)]
    assert evaluation_calls == [("family movie about bears in the woods", 2)]


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
    assert "Reciprocal Rank Fusion Results for 'briish bear' (k=60):" in output
    assert "1. Paddington" in output
    assert search_calls == [("briish bear", 60, 5)]


def test_build_enhancement_prompt_for_rewrite():
    hybrid_search_cli = load_cli_module()

    prompt = hybrid_search_cli.build_enhancement_prompt(
        "movie about bear in london with marmalade", "rewrite"
    )

    assert "Rewrite the user-provided movie search query below" in prompt
    assert 'User query: "movie about bear in london with marmalade"' in prompt


def test_build_enhancement_prompt_for_expand():
    hybrid_search_cli = load_cli_module()

    prompt = hybrid_search_cli.build_enhancement_prompt("math movie", "expand")

    assert "Expand the user-provided movie search query below" in prompt
    assert 'User query: "math movie"' in prompt


def test_build_individual_rerank_prompt():
    hybrid_search_cli = load_cli_module()

    prompt = hybrid_search_cli.build_individual_rerank_prompt(
        "family movie about bears in the woods",
        {"title": "The Country Bears", "document": "A bear searches for belonging."},
    )

    assert 'Query: "family movie about bears in the woods"' in prompt
    assert "Movie: The Country Bears - A bear searches for belonging." in prompt


def test_build_batch_rerank_prompt():
    hybrid_search_cli = load_cli_module()

    prompt = hybrid_search_cli.build_batch_rerank_prompt(
        "family movie about bears in the woods",
        [
            {"title": "The Country Bears", "document": "A bear searches for belonging."},
            {"title": "Goldilocks and the Three Bears", "document": "Three bears in the woods."},
        ],
    )

    assert 'Query: "family movie about bears in the woods"' in prompt
    assert "1: The Country Bears - A bear searches for belonging." in prompt
    assert "2: Goldilocks and the Three Bears - Three bears in the woods." in prompt
    assert "raw JSON array of integers" in prompt


def test_build_cross_encoder_pairs():
    hybrid_search_cli = load_cli_module()

    pairs = hybrid_search_cli.build_cross_encoder_pairs(
        "family movie about bears in the woods",
        [
            {"title": "The Country Bears", "document": "A bear searches for belonging."},
            {"title": "A Bear for Punishment", "document": "A bear family is startled awake."},
        ],
    )

    assert pairs == [
        [
            "family movie about bears in the woods",
            "The Country Bears - A bear searches for belonging.",
        ],
        [
            "family movie about bears in the woods",
            "A Bear for Punishment - A bear family is startled awake.",
        ],
    ]


def test_build_evaluation_prompt():
    hybrid_search_cli = load_cli_module()

    prompt = hybrid_search_cli.build_evaluation_prompt(
        "family movie about bears in the woods",
        [
            {"title": "The Berenstain Bears' Christmas Tree"},
            {"title": "The Country Bears"},
        ],
    )

    assert 'Query: "family movie about bears in the woods"' in prompt
    assert "1. The Berenstain Bears' Christmas Tree" in prompt
    assert "2. The Country Bears" in prompt
    assert "Return ONLY the scores" in prompt
