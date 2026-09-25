"""Phase 1: MongoDB search foundation, exercised through the real repository
code against a seeded in-memory MongoDB (mongomock)."""

from __future__ import annotations

import pytest

from ai_search.app.repository import SearchDocumentsRepository, _token_pattern


def _types(result):
    return {r["entity_type"] for r in result["results"]}


def _titles(result):
    return [r["title"] for r in result["results"]]


def test_chef_jobs_in_dubai(container):
    result = container.search.search(query="chef jobs in Dubai")
    assert result["total"] == 5
    assert _types(result) == {"job"}
    assert all(r["location"]["city"] == "Dubai" for r in result["results"])
    assert all("chef" in r["title"].casefold() for r in result["results"])


def test_hotel_jobs_mumbai(container):
    result = container.search.search(query="hotel jobs Mumbai")
    assert _types(result) == {"job"}
    assert "Hotel Manager" in _titles(result)
    assert all(r["location"]["city"] == "Mumbai" for r in result["results"])


def test_hr_manager_uae(container):
    result = container.search.search(query="HR manager UAE")
    assert result["results"], result
    assert any("HR Manager" in t for t in _titles(result))
    assert all(
        r["location"]["country"] == "United Arab Emirates" for r in result["results"]
    )


def test_restaurant_suppliers(container):
    result = container.search.search(query="restaurant suppliers")
    assert result["understanding"]["clarification"] is None
    assert _types(result) == {"product"}
    assert "Restaurant POS System" in _titles(result)


def test_hospitality_companies(container):
    result = container.search.search(query="hospitality companies")
    assert result["understanding"]["clarification"] is None
    assert _types(result) == {"company"}
    assert {"ABC Hospitality", "Hospitality Solutions LLC"} <= set(_titles(result))


def test_events_dubai_browse(container):
    result = container.search.search(query="events Dubai")
    assert result["understanding"]["browse"] is True
    assert _titles(result) == ["The Hotel Show Dubai 2026"]


def test_hospitality_articles(container):
    result = container.search.search(query="hospitality articles")
    assert _types(result) == {"article"}
    assert "Hospitality trends 2026" in _titles(result)


def test_awards(container):
    result = container.search.search(query="hospitality awards")
    assert _titles(result) == ["Hospitality Excellence Awards 2026"]
    # Entity filtering alone at the repository level.
    docs = container.repository.browse(
        entity="award", city=None, country=None, status=None, is_live=None
    )
    assert {d["entity_type"] for d in docs} == {"award"}


def test_faqs(container):
    result = container.search.search(query="FAQ how to apply for a job")
    assert _types(result) == {"faq"}
    assert result["results"][0]["title"] == "How do I apply for a job?"


def test_all_eight_entity_types_are_searchable(container):
    for entity in (
        "job",
        "professional",
        "company",
        "product",
        "article",
        "event",
        "award",
        "faq",
    ):
        docs = container.repository.browse(
            entity=entity, city=None, country=None, status=None, is_live=None
        )
        assert docs and {d["entity_type"] for d in docs} == {entity}


def test_max_five_results(container):
    result = container.search.search(query="chef", limit=5)
    assert len(result["results"]) <= 5


def test_country_filter_is_hard(container):
    result = container.search.search(query="chef jobs", country="India")
    assert result["results"]
    assert all(r["location"]["country"] == "India" for r in result["results"])


def test_live_filter(container, mongo_db):
    mongo_db["search_documents"].update_one(
        {"_id": "job:102"}, {"$set": {"is_live": False}}
    )
    result = container.search.search(
        query="sous chef", entity="job", city="Dubai", is_live=True
    )
    assert "job:102" not in {f"job:{r['entity_id']}" for r in result["results"]}


def test_status_filter(container):
    result = container.search.search(
        query="chef", entity="job", city="Dubai", status="closed"
    )
    assert result["results"] == []


def test_typo_correction(container):
    result = container.search.search(query="excutive chef jobs in Dubai")
    assert result["corrected_query"] == "executive chef"
    assert result["results"][0]["title"] == "Executive Chef"


def test_result_urls_come_from_records(container):
    result = container.search.search(query="chef jobs in Dubai")
    for item in result["results"]:
        assert item["url"] == f"https://www.hozpitality.com/jobs/{item['entity_id']}"


def test_slug_is_never_returned_as_url():
    from ai_search.app.service import SearchService

    payload = SearchService._result_payload(
        {"title": "X", "slug": "x-job-1", "entity_type": "job"}, 1.0, []
    )
    assert payload["url"] is None


def test_fetch_by_ids_supports_object_ids(container):
    from ai_search.tests.fixtures_data import INJECTION_OID

    docs = container.repository.fetch_by_ids([str(INJECTION_OID), "job:101"])
    assert [str(d["_id"]) for d in docs] == [str(INJECTION_OID), "job:101"]


@pytest.mark.parametrize(
    "token,expected", [("chefs", "chef"), ("companies", "company"), ("chef", None)]
)
def test_plural_tokens_match_singular_titles(token, expected):
    import re

    pattern = _token_pattern(token)
    if expected:
        assert re.search(pattern, f"Executive {expected.title()}", re.I)
    assert re.search(pattern, token, re.I)


def test_required_text_index_is_verified(mongo_db):
    repo = SearchDocumentsRepository(mongo_db["search_documents"])
    repo.ensure_indexes()  # must not raise when idx_ai_search_text exists
    mongo_db["search_documents"].drop_index("idx_ai_search_text")
    with pytest.raises(RuntimeError):
        repo.ensure_indexes()
