import json
from types import SimpleNamespace

from ai_search.app.llm import OllamaQueryInterpreter
from ai_search.app.query_understanding import SearchPlan


def test_ollama_interpreter_parses_structured_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "message": {
                    "content": json.dumps({
                        "intent": "search",
                        "entity": "job",
                        "keywords": ["chef"],
                        "city": "Dubai",
                        "country": "United Arab Emirates",
                        "experience": 5,
                        "level": "senior",
                        "department": None,
                        "industry": None,
                        "category": None,
                        "filters": {}
                    })
                }
            }).encode()

    def fake_urlopen(req, timeout):
        assert "/api/chat" in req.full_url
        payload = json.loads(req.data.decode())
        assert payload["model"] == "qwen3:8b"
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["format"]["type"] == "object"
        return FakeResponse()

    monkeypatch.setattr("ai_search.app.llm.urlrequest.urlopen", fake_urlopen)
    monkeypatch.setattr("ai_search.app.llm.settings", SimpleNamespace(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_query_model="qwen3:8b",
        ollama_timeout_seconds=20,
    ))

    interpreter = OllamaQueryInterpreter()
    plan = interpreter.interpret(
        "I need a senior chef job in Dubai with 5 years experience",
        SearchPlan()
    )

    assert plan.entity == "job"
    assert plan.keywords == ["chef"]
    assert plan.city == "Dubai"
    assert plan.country == "United Arab Emirates"
    assert plan.experience == 5
    assert plan.level == "senior"


def test_ollama_disabled_falls_back_to_deterministic(monkeypatch):
    monkeypatch.setattr("ai_search.app.llm.settings", SimpleNamespace(
        ollama_base_url="",
        ollama_query_model="",
        ollama_timeout_seconds=1,
    ))
    base = SearchPlan(entity="job", keywords=["chef"], confidence=0.9)
    result = OllamaQueryInterpreter().interpret("chef jobs", base)
    assert result is base
