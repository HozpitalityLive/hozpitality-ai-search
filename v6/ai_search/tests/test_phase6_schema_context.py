"""Architecture-level regressions: conversation context transitions, FAQ
intent, schema-correct supplier/industry/category handling, database-only
results and clickable URLs.

The corpus is produced by the real migration document builders
(fixtures_data.py), so these tests exercise the production schema.
"""

from __future__ import annotations

import pytest

from ai_search.app.intent import classify, is_information_question
from ai_search.app.query_understanding import understand
from ai_search.app.schema_map import SCHEMAS, filter_applies

SUPPLIERS = {"company:1005", "company:1006", "company:1007"}
SUPPLIER_INDUSTRY = {
    "company:1005",
    "company:1006",
}  # industries[].context == "supplier"
NOT_A_SUPPLIER = "company:1008"  # industry NAMED "Supplier Relations Consulting", context hospitality


def say(chat, message, cid="conversation-schema-0001"):
    return chat.chat(message=message, conversation_id=cid)


def ids(response, key="results"):
    return [r["id"] for r in response[key]]


def types(response, key="results"):
    return {r["entity_type"] for r in response[key]}


def context(response):
    return response["understanding"]["context"]


# ---------------------------------------------------------------------------
# A. Context switch: professional -> job
# ---------------------------------------------------------------------------


def test_a_new_explicit_job_search_does_not_inherit_professional_context(chat):
    r1 = say(chat, "Find professionals in Dubai")
    assert r1["action"] == "search"
    assert r1["state"]["entity"] == "professional"
    assert types(r1) == {"professional"}

    r2 = say(chat, "Find chef jobs in Dubai")
    assert r2["state"]["entity"] == "job"
    assert r2["state"]["keywords"] == ["chef"]
    assert types(r2) == {"job"}
    assert context(r2)["transition"] == "new_search"
    assert r2["understanding"]["is_new_search"] is True
    assert r2["understanding"]["is_context_continuation"] is False


# ---------------------------------------------------------------------------
# B. Job continuation
# ---------------------------------------------------------------------------


def test_b_job_context_is_kept_through_refinements_and_more(chat):
    r1 = say(chat, "Find chef jobs in Dubai")
    r2 = say(chat, "Only management positions")
    r3 = say(chat, "Only jobs with accommodation")
    r4 = say(chat, "Show me more")
    for r in (r1, r2, r3, r4):
        assert r["state"]["entity"] == "job"
        assert r["state"]["keywords"] == ["chef"]
        assert r["state"]["location"]["city"] == "Dubai"
        assert types(r) <= {"job"}
    assert context(r2)["transition"] == "modification"
    assert context(r3)["transition"] == "modification"
    assert "entity" in context(r3)["inherited"]
    assert r3["state"]["filters"] == {"level": "manager", "accommodation": True}
    assert context(r4)["transition"] == "continuation"
    assert not set(ids(r4)) & set(ids(r3))


# ---------------------------------------------------------------------------
# C. Professional continuation
# ---------------------------------------------------------------------------


def test_c_professional_context_is_kept(chat):
    r1 = say(chat, "Find professionals in Dubai")
    r2 = say(chat, "Only executive professionals")
    r3 = say(chat, "Show me more")
    assert [r["state"]["entity"] for r in (r1, r2, r3)] == ["professional"] * 3
    assert context(r2)["transition"] == "modification"
    assert r2["state"]["filters"] == {"level": "executive"}
    assert types(r2) == {"professional"}
    # Evidence from professional.job_level / job_role (Executive Housekeeper, Executive Chef)
    assert {r["title"] for r in r2["results"]} == {"Sara Ali", "Ahmed Khan"}
    assert types(r3, "related_results") <= {"professional"}


# ---------------------------------------------------------------------------
# D. Entity switch resets incompatible filters
# ---------------------------------------------------------------------------


def test_d_new_company_search_drops_job_filters_and_keywords(chat):
    say(chat, "Find chef jobs in Dubai")
    say(chat, "Only management positions")
    say(chat, "With accommodation")
    r = say(chat, "Find companies in Dubai")
    assert r["state"]["entity"] == "company"
    assert r["state"]["filters"] == {}
    assert r["state"]["keywords"] == []  # no "chef companies"
    assert r["state"]["location"]["city"] == "Dubai"  # stated in the new message
    assert types(r) == {"company"}
    ctx = context(r)
    assert ctx["transition"] == "new_search"
    assert {"keywords", "filters"} <= set(ctx["discarded"])


def test_entity_switch_with_context_keeps_only_compatible_state(chat):
    say(chat, "Find chef jobs in Dubai")
    say(chat, "Only management positions")
    say(chat, "With accommodation")
    r = say(chat, "Actually show professionals instead")
    assert r["state"]["entity"] == "professional"
    assert r["state"]["keywords"] == [
        "chef"
    ]  # people keywords transfer job <-> professional
    assert r["state"]["location"]["city"] == "Dubai"
    assert "accommodation" not in r["state"]["filters"]  # job-only filter
    assert r["state"]["strict_filters"] == []
    assert context(r)["transition"] == "entity_switch"
    assert "filters.accommodation" in context(r)["discarded"]
    assert types(r) == {"professional"}


def test_stale_job_filters_never_reach_other_entities(chat):
    say(chat, "Find chef jobs in Dubai")
    say(chat, "Only management positions")
    say(chat, "With accommodation")
    for message, entity in (
        ("Find professionals in Dubai", "professional"),
        ("Find articles about hotel technology", "article"),
        ("Find events in Dubai", "event"),
        ("Find hospitality awards", "award"),
        ("Find hotel products in Dubai", "product"),
    ):
        r = say(chat, message)
        assert r["state"]["entity"] == entity, message
        assert "accommodation" not in r["state"]["filters"], message
        assert "level" not in r["state"]["filters"], message
        assert types(r) <= {entity}, message


def test_spec_section_12_multiple_independent_searches(chat):
    say(chat, "Find chef jobs in Dubai")
    say(chat, "Only management positions")
    r3 = say(chat, "Show me more")
    assert r3["state"]["entity"] == "job"
    r4 = say(chat, "Find professionals in Mumbai")
    assert r4["state"]["entity"] == "professional" and r4["state"]["filters"] == {}
    r5 = say(chat, "Show me more")
    assert r5["state"]["entity"] == "professional"
    assert types(r5) | types(r5, "related_results") <= {"professional"}
    r6 = say(chat, "Find companies in Dubai")
    assert r6["state"]["entity"] == "company" and types(r6) == {"company"}


# ---------------------------------------------------------------------------
# E/F. FAQ intent and database-backed answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "How to apply for a job?",
        "How do I apply for a job?",
        "How can I apply for a job?",
        "How do I apply for a chef job?",
    ],
)
def test_e_apply_questions_are_faq_not_job_search(chat, question):
    r = say(chat, question)
    assert r["action"] == "faq"
    assert "What type of job" not in r["answer"]
    assert "Which location" not in r["answer"]
    assert r["results"][0]["id"] == "faq:1"
    # Verbatim answer from the FAQ record (base_faq.answer).
    assert "Open the job listing and click Apply Now." in r["answer"]
    assert r["understanding"]["is_faq"] is True


def test_f_application_process_is_faq(chat):
    r = say(chat, "What is the application process?")
    assert r["action"] == "faq"
    assert r["results"][0]["id"] == "faq:6"
    assert "employer reviews your profile" in r["answer"]


def test_faq_without_a_matching_record_is_not_invented(chat):
    r = say(chat, "What is the leave policy?")
    assert r["action"] == "faq"
    assert r["results"] == []
    assert "couldn't find an answer" in r["answer"]
    assert "leave" not in r["answer"].split("couldn't")[0]


def test_faq_answers_do_not_call_the_llm(container):
    import httpx

    from ai_search.app.chat_service import ChatService
    from ai_search.app.ollama_client import OllamaChatClient

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            200, json={"message": {"content": "invented"}, "done": True}
        )

    llm = OllamaChatClient(
        base_url="http://ollama.test",
        model="qwen3:8b",
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    chat = ChatService(container.search, container.conversations, llm)
    r = chat.chat(
        message="How can I reset my password?", conversation_id="conversation-faq-0001"
    )
    assert calls == []
    assert r["answer"].endswith(
        "Use Forgot Password on the login page to receive a reset link by email."
    )


@pytest.mark.parametrize(
    "message,expected",
    [
        ("How do I apply for a job?", True),
        ("What is the application process?", True),
        ("How does the job application work?", True),
        ("How do I create an account?", True),
        ("How can I reset my password?", True),
        ("What is Hozpitality?", True),
        ("How does this work?", True),
        ("What are the requirements?", True),
        ("How can I contact support?", True),
        ("What is the policy?", True),
        ("How do I use this feature?", True),
        ("Why can't I apply for a job?", True),
        ("What is the application process for jobs?", True),
        ("Find chef jobs in Dubai", False),
        ("Which chef jobs are available in Dubai?", False),
        ("What chef jobs are available?", False),
        ("are there any chef jobs in Dubai", False),
        ("Find FAQs about passwords", False),  # a search of the FAQ module
    ],
)
def test_information_vs_records_detection(message, expected):
    assert is_information_question(message)[0] is expected


# ---------------------------------------------------------------------------
# G/H. Explicit entity words beat role words
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,entity",
    [
        ("chef jobs", "job"),
        ("chef job openings", "job"),
        ("find a chef position", "job"),
        ("chef professionals", "professional"),
        ("find professionals who are chefs", "professional"),
        ("chef companies", "company"),
        ("chef articles", "article"),
        ("chef awards", "award"),
        ("chef events", "event"),
        ("chef suppliers", "company"),
        ("how to apply for a job", "faq"),
    ],
)
def test_explicit_entity_priority(message, entity):
    assert understand(message).entity == entity


def test_g_job_search(chat):
    r = say(chat, "Find chef jobs in Dubai")
    assert r["state"]["entity"] == "job" and types(r) == {"job"}


def test_h_professional_search(chat):
    r = say(chat, "Find chef professionals in Dubai")
    assert r["state"]["entity"] == "professional" and types(r) == {"professional"}
    assert all(r_["location"]["city"] == "Dubai" for r_ in r["results"])


def test_llm_fallback_cannot_override_explicit_entity(container, monkeypatch):
    def fake_interpret(query, base):
        base.entity = "professional"  # the model "associates chef with people"
        base.confidence = 0.9
        return base

    monkeypatch.setattr(container.search.llm, "interpret", fake_interpret)
    monkeypatch.setattr(
        type(container.search.llm), "enabled", property(lambda self: True)
    )
    result = container.search.search(
        query="chef jobs"
    )  # low confidence -> LLM consulted
    assert result["understanding"]["entity"] == "job"


# ---------------------------------------------------------------------------
# I/J/K. Supplier vs company vs industry vs category (migrate_companies.py)
# ---------------------------------------------------------------------------


def test_i_suppliers_are_companies_with_is_supplier(chat, mongo_db):
    r = say(chat, "Find suppliers")
    assert r["state"]["entity"] == "company"
    assert r["state"]["filters"] == {"is_supplier": True}
    assert set(ids(r)) == SUPPLIERS
    assert NOT_A_SUPPLIER not in ids(
        r
    )  # industry *named* "Supplier ..." is not a supplier
    for item in r["results"]:
        doc = mongo_db["search_documents"].find_one({"_id": item["id"]})
        assert doc["company"]["is_supplier"] is True


def test_j_supplier_industry_filters_industry_context(chat):
    r = say(chat, "Find companies in supplier industry")
    assert r["state"]["entity"] == "company"
    assert r["state"]["filters"] == {"industry_context": "supplier"}
    assert r["state"]["location"]["city"] is None  # "supplier industry" is not a place
    assert set(ids(r)) == SUPPLIER_INDUSTRY
    assert "company:1007" not in ids(r)  # supplier via category only, not industry


def test_k_supplier_category_lists_real_categories(chat):
    r = say(chat, "Find supplier category")
    assert r["action"] == "facet"
    assert r["facets"] == [
        {"name": "Kitchen Equipment", "count": 1},
        {"name": "Linen & Textiles", "count": 1},
    ]
    assert r["results"] == []
    r2 = say(chat, "Find suppliers in supplier category Kitchen Equipment")
    assert r2["state"]["entity"] == "company"
    assert ids(r2) == ["company:1006"]
    # The consumed category value is a filter, not a keyword/department/place.
    assert r2["state"]["keywords"] == []
    assert r2["state"]["filters"] == {
        "is_supplier": True,
        "supplier_category": "kitchen equipment",
    }
    assert r2["state"]["location"]["city"] is None


def test_marketplace_products_are_not_suppliers(chat):
    r = say(chat, "Find products from suppliers")
    assert r["state"]["entity"] == "product"
    assert types(r) <= {"product"}
    r2 = say(chat, "restaurant suppliers")
    assert types(r2) == {"company"}


def test_companies_hiring_uses_the_job_company_relationship(chat, mongo_db):
    r = say(chat, "companies hiring chefs in Dubai")
    assert types(r) == {"company"}
    for item in r["results"]:
        company_id = int(item["entity_id"])
        # Each company actually owns a matching live chef job in Dubai.
        job = mongo_db["search_documents"].find_one(
            {"entity_type": "job", "company.id": company_id, "location.city": "Dubai"}
        )
        assert job is not None
        assert item["metadata"]["hiring_for"]


# ---------------------------------------------------------------------------
# L. Database-only results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Find chef jobs in Dubai",
        "Find professionals in Dubai",
        "Find suppliers",
        "hotel companies in Dubai",
        "articles about hotel technology",
        "events in Dubai",
        "hospitality awards",
        "How to apply for a job?",
        "Find hotel products in Dubai",
        "companies hiring chefs in Dubai",
        "suppliers in Antarctica",
    ],
)
def test_l_every_result_is_a_real_record(chat, mongo_db, message):
    r = say(chat, message, cid="conversation-dbonly-01")
    for item in r["results"] + r["related_results"]:
        doc = mongo_db["search_documents"].find_one({"_id": item["id"]})
        assert doc is not None, item["id"]
        assert doc["entity_type"] == item["entity_type"]
        assert item["title"] in {
            doc.get("title"),
            doc.get("question"),
            str(doc.get("question", "")).split(". ", 1)[-1],
        }


# ---------------------------------------------------------------------------
# M. Clickable URLs from real data only
# ---------------------------------------------------------------------------


def test_m_urls_are_real_or_absent(chat):
    r = say(chat, "hospitality awards")
    award = next(x for x in r["results"] if x["id"] == "award:1")
    assert (
        award["url"].startswith("https://www.hozpitality.com/awards/")
        and award["url_source"] == "record"
    )
    jobs = say(chat, "Find chef jobs in Dubai")
    assert all(
        x["url"] is None for x in jobs["results"]
    )  # no route template configured


def test_m_urls_from_configured_templates(chat, url_templates):
    r = say(chat, "Find chef jobs in Dubai")
    for item in r["results"]:
        assert item["url"] == f"https://www.hozpitality.com/test-jobs/{item['slug']}"
        assert item["company_ref"]["url"].startswith(
            "https://www.hozpitality.com/test-companies/"
        )
    # Articles have no template configured -> still no URL.
    a = say(chat, "articles about hotel technology")
    assert all(x["url"] is None for x in a["results"])


def test_external_links_are_labelled_and_not_used_as_page_urls(chat):
    r = say(chat, "sous chef night shift jobs in Dubai")
    item = next(x for x in r["results"] if x["id"] == "job:999")
    assert item["url"] is None
    assert item["external_url"] == "https://careers.example.com/listing/999"
    assert item["external_label"] == "Original listing"


# ---------------------------------------------------------------------------
# N. No results are never replaced by another location/entity
# ---------------------------------------------------------------------------


def test_n_unknown_location_keeps_entity_and_labels_related(chat):
    r = say(chat, "suppliers in Antarctica")
    assert r["results"] == []
    assert r["related_results"]
    assert (
        set(ids(r, "related_results")) <= SUPPLIERS
    )  # relaxed location, never relaxed entity
    assert "Antarctica" in r["answer"] and "couldn't find an exact match" in r["answer"]


def test_n_unknown_topic_returns_nothing(chat):
    r = say(chat, "Find spaceship pilot jobs in Dubai")
    assert r["results"] == []
    assert types(r, "related_results") <= {"job"}


# ---------------------------------------------------------------------------
# O. Comparison in the current entity context
# ---------------------------------------------------------------------------


def test_o_compare_first_three_professionals(chat):
    say(chat, "Find chef jobs in Dubai")
    r1 = say(chat, "Find professionals in Dubai")
    r = say(chat, "Compare the first three")
    assert r["action"] == "compare"
    assert ids(r) == ids(r1)[:3]
    assert types(r) == {"professional"}


# ---------------------------------------------------------------------------
# Schema map and /search/understand
# ---------------------------------------------------------------------------


def test_schema_map_covers_all_modules_and_filter_lifetime():
    assert set(SCHEMAS) == {
        "job",
        "professional",
        "company",
        "product",
        "article",
        "event",
        "award",
        "faq",
    }
    assert filter_applies("job", "level") and filter_applies("professional", "level")
    assert not filter_applies("company", "level")
    assert filter_applies("job", "accommodation") and not filter_applies(
        "professional", "accommodation"
    )
    assert filter_applies("company", "is_supplier") and not filter_applies(
        "job", "is_supplier"
    )


def test_fixture_corpus_matches_migration_shapes(mongo_db):
    for doc in mongo_db["search_documents"].find():
        assert doc["_id"].startswith(f"{doc['entity_type']}:")
        assert "ai_search_text" in doc or doc["entity_type"] == "faq"


def test_classification_reasons_are_explainable():
    assert "company.is_supplier" in classify("Find suppliers").reason
    assert classify("Find supplier category").facet == "supplier_category"
    assert classify("companies hiring chefs").related == "companies_hiring"


def test_search_understand_endpoint(client):
    data = client.get(
        "/search/understand", params={"q": "Find chef jobs in Dubai"}
    ).json()
    assert data["entity"] == "job" and data["keywords"] == ["chef"]
    assert data["location"]["city"] == "Dubai"
    assert data["is_faq"] is False and data["is_new_search"] is True
    assert "jobs" in data["entity_reason"]
    faq = client.get(
        "/search/understand", params={"q": "How to apply for a job?"}
    ).json()
    assert (
        faq["is_faq"] is True
        and faq["entity"] == "faq"
        and faq["clarification"] is None
    )


def test_search_understand_dry_run_against_a_conversation(client):
    cid = "conversation-dryrun-001"
    client.post(
        "/chat", json={"message": "Find chef jobs in Dubai", "conversation_id": cid}
    )
    client.post(
        "/chat", json={"message": "Only management positions", "conversation_id": cid}
    )
    data = client.get(
        "/search/understand",
        params={"q": "Find companies in Dubai", "conversation_id": cid},
    ).json()
    assert data["context"]["transition"] == "new_search"
    assert "filters" in data["context"]["discarded"]
    assert data["resulting_state"]["entity"] == "company"
    assert data["resulting_state"]["filters"] == {}
    # Nothing was saved by the dry run.
    assert client.get(f"/chat/{cid}").json()["state"]["entity"] == "job"
    cont = client.get(
        "/search/understand", params={"q": "With accommodation", "conversation_id": cid}
    ).json()
    assert cont["is_context_continuation"] is True
    assert cont["resulting_state"]["filters"] == {
        "level": "manager",
        "accommodation": True,
    }


def test_information_reasons_are_readable():
    from ai_search.app import intent as intent_module

    assert len(intent_module.INFO_QUESTION_LABELS) == len(
        intent_module.INFO_QUESTION_RES
    )
    assert (
        is_information_question("How to apply for a job?")[1]
        == "information question: how-to question"
    )
