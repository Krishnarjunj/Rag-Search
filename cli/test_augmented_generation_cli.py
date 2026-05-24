import importlib
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
    sys.modules.pop("augmented_generation_cli", None)
    return importlib.import_module("augmented_generation_cli")


def test_build_rag_prompt():
    cli = load_cli_module()

    prompt = cli.build_rag_prompt("movies about action and dinosaurs", "Jurassic Park: dinosaur park")

    assert "You are a RAG agent for Hoopla" in prompt
    assert "Query: movies about action and dinosaurs" in prompt
    assert "Documents:" in prompt
    assert "Jurassic Park: dinosaur park" in prompt


def test_rag_command_prints_search_results_and_response(monkeypatch, capsys):
    cli = load_cli_module()

    class FakeHybridSearch:
        def __init__(self, documents):
            self.documents = documents

        def rrf_search(self, query, k, limit):
            assert query == "movies about action and dinosaurs"
            assert k == 60
            assert limit == 5
            return [
                {"title": "We're Back! A Dinosaur's Story", "document": "Dinosaurs in New York."},
                {"title": "Jurassic Park", "document": "Dinosaurs escape a theme park."},
            ]

    class FakeResponse:
        text = "Use Jurassic Park for action-dinosaur recommendations."

    class FakeModels:
        def generate_content(self, model, contents):
            assert "Query: movies about action and dinosaurs" in contents
            assert "Jurassic Park: Dinosaurs escape a theme park." in contents
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_google.genai)
    monkeypatch.setattr(cli, "HybridSearch", FakeHybridSearch)
    monkeypatch.setattr(cli, "load_movies", lambda: [{"title": "Jurassic Park"}])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        sys,
        "argv",
        ["augmented_generation_cli.py", "rag", "movies about action and dinosaurs"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Search Results:" in output
    assert "Jurassic Park" in output
    assert "RAG Response:" in output
    assert "Use Jurassic Park for action-dinosaur recommendations." in output
