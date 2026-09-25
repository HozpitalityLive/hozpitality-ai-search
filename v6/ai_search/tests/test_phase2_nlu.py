"""Phase 2: natural-language understanding + smart search (required queries)."""

from __future__ import annotations

from ai_search.app.query_understanding import clarification_for, understand


def test_senior_chef_job_with_experience(container):
    result = container.search.search(
        query="I need a senior chef job in Dubai with 5 years experience"
    )
    u = result["understanding"]
    assert u["intent"] == "search"
    assert u["entity"] == "job"
    assert u["keywords"] == ["chef"]
    assert u["city"] == "Dubai"
    assert u["country"] == "United Arab Emirates"
    assert u["experience"] == 5
    assert u["level"] == "senior"
    assert {r["entity_type"] for r in result["results"]} == {"job"}


def test_misspelled_executive(container):
    result = container.search.search(query="excutive chef jobs in Dubai")
    assert result["corrected_query"] == "executive chef"
    assert result["understanding"]["corrections"][0]["to"] == "executive"


def test_find_me_a_job_asks_for_type(container):
    result = container.search.search(query="Find me a job")
    assert (
        result["understanding"]["clarification"]
        == "What type of job are you looking for?"
    )
    assert result["results"] == []


def test_find_chef_jobs_asks_for_location(container):
    result = container.search.search(query="Find chef jobs")
    assert (
        result["understanding"]["clarification"] == "Which location would you prefer?"
    )


def test_find_chef_jobs_in_dubai_searches_directly(container):
    result = container.search.search(query="Find chef jobs in Dubai")
    assert result["understanding"]["clarification"] is None
    assert result["total"] == 5


def test_find_senior_chefs_in_dubai_is_professional(container):
    result = container.search.search(query="find senior chefs in Dubai")
    assert result["understanding"]["entity"] == "professional"
    assert {r["entity_type"] for r in result["results"]} == {"professional"}


def test_hotel_companies_in_dubai(container):
    result = container.search.search(query="hotel companies in Dubai")
    assert result["understanding"]["entity"] == "company"
    assert {r["title"] for r in result["results"]} == {
        "ABC Hospitality",
        "XYZ Hotels",
        "Marina Resorts",
    }


def test_articles_about_hotel_technology(container):
    result = container.search.search(query="articles about hotel technology")
    assert result["understanding"]["clarification"] is None
    assert (
        result["results"][0]["title"]
        == "How hotel technology is transforming guest experience"
    )


def test_hotel_products_in_dubai(container):
    result = container.search.search(query="hotel products in Dubai")
    assert [r["title"] for r in result["results"]] == ["Hotel Linen Supplies"]


def test_quantum_chef_jobs_in_antarctica(container):
    result = container.search.search(query="quantum chef jobs in Antarctica")
    assert result["understanding"]["clarification"] is None
    assert result["understanding"]["explicit_location"] is True
    assert result["results"] == []


def test_chef_jobs_in_antarctica_returns_related_only(container):
    result = container.search.search(query="chef jobs in Antarctica")
    assert result["results"] == []
    assert result["related_results"]
    assert all("related_result" in r["matched_by"] for r in result["related_results"])
    assert result["message"].startswith("I couldn't find an exact match")


def test_long_natural_language_professional_query(container):
    query = "I am looking for an experienced executive chef who can manage a luxury hotel kitchen in Dubai"
    plan = understand(query)
    assert plan.entity == "professional"
    assert {"executive", "chef"} <= set(plan.keywords)
    assert "am" not in plan.keywords and "who" not in plan.keywords
    result = container.search.search(query=query)
    assert result["results"][0]["title"] == "Ahmed K. - Executive Chef"


def test_bare_keyword_is_not_blocked_by_clarification():
    # Phase 1 contract: /search?q=chef searches; an entity inferred from a
    # role word must not demand a location.
    plan = understand("chef")
    assert plan.entity == "professional" and plan.entity_source == "role"
    assert clarification_for(plan) is None


def test_entity_source_is_recorded():
    assert understand("chef jobs").entity_source == "text"
    assert understand("hiring waiters").entity_source == "intent"
    assert understand("waiters").entity_source == "role"


def test_semantic_scores_are_merged_as_ranking_signal(container):
    class FakeSemantic:
        enabled = True

        def search(self, query, limit=50):
            return [("job:105", 0.9)]

    container.search.semantic = FakeSemantic()
    result = container.search.search(query="chef jobs in Dubai")
    top = result["results"][0]
    assert top["entity_id"] == "105"
    assert "semantic" in top["matched_by"]


def test_llm_query_filters_are_accepted_only_when_stated(monkeypatch):
    """Regression: the allow-list regexes used a literal backslash and never matched."""
    import json
    from types import SimpleNamespace

    from ai_search.app.llm import OllamaQueryInterpreter
    from ai_search.app.query_understanding import SearchPlan

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            content = {
                "intent": "search",
                "entity": "job",
                "keywords": ["chef"],
                "city": None,
                "country": None,
                "experience": None,
                "level": None,
                "department": None,
                "industry": None,
                "category": None,
                "filters": {"employment_type": "full-time", "verified": True},
            }
            return json.dumps({"message": {"content": json.dumps(content)}}).encode()

    monkeypatch.setattr(
        "ai_search.app.llm.urlrequest.urlopen", lambda req, timeout: FakeResponse()
    )
    monkeypatch.setattr(
        "ai_search.app.llm.settings",
        SimpleNamespace(
            ollama_base_url="http://ollama",
            ollama_query_model="qwen3:8b",
            ollama_timeout_seconds=60,
            ollama_query_timeout_seconds=5,
        ),
    )
    plan = OllamaQueryInterpreter().interpret("full-time chef work", SearchPlan())
    assert plan.filters == {
        "employment_type": "full-time"
    }  # "verified" was never stated
