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
    # Professionals' titles are names (migrate_professionals); the role is metadata.
    assert any(r["metadata"].get("role") == "HR Manager" for r in result["results"])
    assert all(
        r["location"]["country"] == "United Arab Emirates" for r in result["results"]
    )


def test_restaurant_suppliers(container):
    # Per migrate_companies.py suppliers are companies with company.is_supplier;
    # marketplace products are a different module.
    result = container.search.search(query="restaurant suppliers")
    assert result["understanding"]["clarification"] is None
    assert _types(result) == {"company"}
    assert _titles(result) == ["KitchenPro Supplies"]
    assert all(r["metadata"]["is_supplier"] for r in result["results"])


def test_hospitality_companies(container):
    result = container.search.search(query="hospitality companies")
    assert result["understanding"]["clarification"] is None
    assert _types(result) == {"company"}
    assert {"ABC Hospitality", "Hospitality Solutions LLC"} <= set(_titles(result))


def test_events_dubai_browse(container):
    result = container.search.search(query="events Dubai")
    assert result["understanding"]["browse"] is True
    assert set(_titles(result)) == {"The Hotel Show Dubai 2026", "Chef Culinary Expo"}
    assert all(r["location"]["city"] == "Dubai" for r in result["results"])


def test_hospitality_articles(container):
    result = container.search.search(query="hospitality articles")
    assert _types(result) == {"article"}
    assert "Hospitality trends 2026" in _titles(result)


def test_awards(container):
    result = container.search.search(query="hospitality awards")
    assert _types(result) == {"award"}
    assert _titles(result)[0] == "Hospitality Excellence Awards 2026"
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
    # migrate_jobs writes liveness twice (is_live and metadata.is_live).
    mongo_db["search_documents"].update_one(
        {"_id": "job:102"}, {"$set": {"is_live": False, "metadata.is_live": False}}
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


def test_no_url_is_fabricated_without_a_route_template(container):
    # Jobs have a slug but no URL in the migration: without a configured route
    # template the backend must return url=None (never a guessed URL).
    result = container.search.search(query="chef jobs in Dubai")
    assert result["results"]
    for item in result["results"]:
        assert item["url"] is None and item["url_source"] is None
        assert item["slug"]


def test_urls_are_built_from_configured_templates_and_real_slugs(
    container, url_templates
):
    result = container.search.search(query="chef jobs in Dubai")
    for item in result["results"]:
        assert item["url"] == f"https://www.hozpitality.com/test-jobs/{item['slug']}"
        assert item["url_source"] == "template"
        company = item["company_ref"]
        assert (
            company["url"]
            == f"https://www.hozpitality.com/test-companies/{company['slug']}"
        )


def test_award_url_comes_from_the_record(container):
    from ai_search.tests.fixtures_data import AWARD_URL

    result = container.search.search(query="hospitality excellence awards")
    top = result["results"][0]
    assert top["url"] == AWARD_URL and top["url_source"] == "record"
    no_url = container.search.search(query="chef of the year awards")["results"][0]
    assert no_url["title"] == "Chef of the Year Awards"
    assert no_url["url"] is None  # award_detail_url is NULL for this award


def test_slug_is_never_returned_as_url():
    from ai_search.app.service import SearchService

    payload = SearchService._result_payload(
        {"title": "X", "slug": "x-job-1", "entity_type": "job"}, 1.0, []
    )
    assert payload["url"] is None


def test_fetch_by_ids_supports_object_ids(container, mongo_db):
    from bson import ObjectId

    oid = ObjectId("65f000000000000000000001")
    mongo_db["search_documents"].insert_one(
        {"_id": oid, "entity_type": "job", "title": "Legacy"}
    )
    docs = container.repository.fetch_by_ids([str(oid), "job:101"])
    assert [str(d["_id"]) for d in docs] == [str(oid), "job:101"]


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
