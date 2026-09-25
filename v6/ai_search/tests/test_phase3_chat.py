"""Phase 3: conversation orchestration, memory, follow-ups, references.

Runs the real ChatService + SearchService + repository + conversation store
over an in-memory MongoDB seeded with realistic documents.
"""

from __future__ import annotations

import pytest

from ai_search.app.chat_service import ChatService
from ai_search.app.dialogue import interpret
from ai_search.app.query_understanding import understand
from ai_search.app.state import apply_intent, empty_state, normalize_state, to_plan

CID = "conversation-test-0001"


def say(chat, message, cid=CID):
    return chat.chat(message=message, conversation_id=cid)


def job_ids(response, key="results"):
    return [f"{r['entity_type']}:{r['entity_id']}" for r in response[key]]


# ---------------------------------------------------------------------------
# Original prototype intents, preserved against the new state API
# ---------------------------------------------------------------------------


def test_accommodation_filter_is_understood():
    plan = understand("chef jobs in Dubai with accommodation")
    assert plan.entity == "job"
    assert plan.city == "Dubai"
    assert plan.filters["accommodation"] is True
    assert "accommodation" not in plan.keywords


def test_follow_up_state_preserves_search_context():
    state = apply_intent(empty_state(), interpret("chef jobs in Dubai", empty_state()))
    state = apply_intent(state, interpret("only management positions", state))
    assert state["entity"] == "job"
    assert state["keywords"] == ["chef"]
    assert state["location"]["city"] == "Dubai"
    assert state["filters"]["level"] == "manager"
    plan = to_plan(state)
    assert (
        plan.keywords == ["chef"] and plan.level == "manager" and plan.entity == "job"
    )


def test_accommodation_follow_up_merges_with_existing_state():
    state = apply_intent(empty_state(), interpret("chef jobs in Dubai", empty_state()))
    state = apply_intent(state, interpret("with accommodation", state))
    assert state["entity"] == "job"
    assert state["keywords"] == ["chef"]
    assert state["location"]["city"] == "Dubai"
    assert state["filters"]["accommodation"] is True
    # Accommodation is a hard, evidence-checked constraint (not a keyword).
    plan = to_plan(state)
    assert plan.filters["accommodation"] is True
    assert (
        "accommodation" in plan.strict_filters
        or state["filters"]["accommodation"] is True
    )


def test_chat_action_detection():
    assert ChatService._action("Show me more") == "more"
    assert ChatService._action("Compare the first three") == "compare"
    assert ChatService._action("Start over") == "reset"
    assert ChatService._action("Only management positions") == "search"


def test_chat_follow_up_keeps_job_entity_in_search(chat, monkeypatch):
    calls = []
    original = chat.search.execute_plan

    def spy(plan, **kwargs):
        calls.append(plan)
        return original(plan, **kwargs)

    monkeypatch.setattr(chat.search, "execute_plan", spy)
    say(chat, "Find chef jobs in Dubai")
    say(chat, "Only management positions")
    assert calls[-1].entity == "job"
    assert calls[-1].city == "Dubai"
    assert "chef" in calls[-1].keywords
    assert calls[-1].level == "manager"


# ---------------------------------------------------------------------------
# CRITICAL REGRESSION TEST
# ---------------------------------------------------------------------------


def test_critical_regression_conversation(chat):
    # Turn 1
    r1 = say(chat, "Find chef jobs in Dubai")
    s1 = r1["state"]
    assert r1["action"] == "search"
    assert s1["entity"] == "job"
    assert s1["location"]["city"] == "Dubai"
    assert s1["keywords"] == ["chef"]
    assert r1["results"] and {r["entity_type"] for r in r1["results"]} == {"job"}

    # Turn 2 - "chef" must NOT flip the entity to professional.
    r2 = say(chat, "Only management positions")
    s2 = r2["state"]
    assert s2["entity"] == "job"
    assert s2["location"]["city"] == "Dubai"
    assert s2["keywords"] == ["chef"]
    assert s2["filters"]["level"] == "manager"
    assert r2["results"] and {r["entity_type"] for r in r2["results"]} == {"job"}
    management_titles = {
        "Executive Chef",
        "Head Chef",
        "Chef Kitchen Manager",
        "Chef de Cuisine",
        "Executive Sous Chef",
        "Head Chef - Banquets",
        "Chef Manager",
    }
    assert {r["title"] for r in r2["results"]} <= management_titles

    # Turn 3
    r3 = say(chat, "With accommodation")
    s3 = r3["state"]
    assert s3["entity"] == "job"
    assert s3["location"]["city"] == "Dubai"
    assert s3["keywords"] == ["chef"]
    assert s3["filters"]["level"] == "manager"
    assert s3["filters"]["accommodation"] is True
    assert r3["results"] and {r["entity_type"] for r in r3["results"]} == {"job"}
    # Only records with real accommodation evidence; 110 says "not provided".
    assert "job:110" not in job_ids(r3)
    assert "job:104" not in job_ids(r3)  # accommodation but entry level

    # Turn 4 - new IDs never displayed before in this search.
    displayed = set(job_ids(r1)) | set(job_ids(r2)) | set(job_ids(r3))
    r4 = say(chat, "Show me more")
    assert r4["action"] == "more"
    assert r4["results"], r4["answer"]
    assert {r["entity_type"] for r in r4["results"]} == {"job"}
    assert not set(job_ids(r4)) & displayed
    assert r4["results"][0]["number"] == len(r3["results"]) + 1  # numbering continues

    # Turn 5 - the actual first three jobs, compared with real fields.
    r5 = say(chat, "Compare the first three")
    assert r5["action"] == "compare"
    assert job_ids(r5) == job_ids(r3)[:3]
    comparison = r5["comparison"]
    assert len(comparison["items"]) == 3
    labels = {f["label"] for f in comparison["fields"]}
    assert {"Company", "Location", "Accommodation"} <= labels
    for item, result in zip(comparison["items"], r3["results"][:3]):
        assert item["values"]["company"] == result["company"]
        assert item["values"]["accommodation"] == "Provided"


def test_acceptance_conversation_continues_with_entity_and_location_changes(chat):
    for message in (
        "Find chef jobs in Dubai",
        "Only management positions",
        "With accommodation",
        "Show me more",
        "Compare the first three",
    ):
        say(chat, message)

    r6 = say(chat, "Actually show professionals instead")
    assert r6["state"]["entity"] == "professional"
    assert r6["state"]["keywords"] == ["chef"]
    assert r6["state"]["location"]["city"] == "Dubai"
    assert "accommodation" not in r6["state"]["filters"]  # job-only filter dropped
    assert r6["results"] and {r["entity_type"] for r in r6["results"]} == {
        "professional"
    }

    r7 = say(chat, "Show me chefs in Abu Dhabi")
    assert r7["state"]["entity"] == "professional"
    assert r7["state"]["location"]["city"] == "Abu Dhabi"
    assert {r["entity_type"] for r in r7["results"]} == {"professional"}
    assert all(r["location"]["city"] == "Abu Dhabi" for r in r7["results"])

    r8 = say(chat, "Start over")
    assert r8["action"] == "reset"
    assert r8["state"]["entity"] is None
    assert r8["state"]["keywords"] == []
    assert r8["state"]["filters"] == {}
    assert r8["state"]["last_results"] == []


# ---------------------------------------------------------------------------
# Location changes and filter removal
# ---------------------------------------------------------------------------


def test_actually_abu_dhabi(chat):
    say(chat, "Find chef jobs in Dubai")
    r = say(chat, "Actually Abu Dhabi")
    assert r["state"]["entity"] == "job"
    assert r["state"]["location"] == {
        "city": "Abu Dhabi",
        "country": "United Arab Emirates",
    }
    assert r["results"] and all(
        x["location"]["city"] == "Abu Dhabi" for x in r["results"]
    )


def test_change_that_to_mumbai(chat):
    say(chat, "Find chef jobs in Dubai")
    r = say(chat, "Change that to Mumbai")
    assert r["state"]["location"]["city"] == "Mumbai"
    assert all(x["location"]["city"] == "Mumbai" for x in r["results"])


def test_remove_the_location(chat):
    say(chat, "Find chef jobs in Dubai")
    r = say(chat, "Remove the location")
    assert r["action"] == "search"
    assert (
        r["state"]["location"]["city"] is None
        and r["state"]["location"]["country"] is None
    )
    assert {x["location"]["city"] for x in r["results"]} - {
        "Dubai"
    }  # other cities appear


def test_anywhere_not_just_dubai(chat):
    say(chat, "Find chef jobs in Dubai")
    r = say(chat, "Anywhere, not just Dubai")
    assert r["state"]["location"]["city"] is None


def test_anywhere_in_uae(chat):
    say(chat, "Find chef jobs in Dubai")
    r = say(chat, "Anywhere in UAE")
    assert r["state"]["location"] == {"city": None, "country": "United Arab Emirates"}
    assert all(x["location"]["country"] == "United Arab Emirates" for x in r["results"])


def test_remove_accommodation_requirement(chat):
    say(chat, "Find chef jobs in Dubai")
    say(chat, "With accommodation")
    r = say(chat, "Remove the accommodation requirement")
    assert "accommodation" not in r["state"]["filters"]
    assert r["state"]["entity"] == "job" and r["state"]["location"]["city"] == "Dubai"


def test_dont_restrict_to_management(chat):
    say(chat, "Find chef jobs in Dubai")
    say(chat, "Only management positions")
    r = say(chat, "Don't restrict it to management")
    assert "level" not in r["state"]["filters"]
    assert r["state"]["keywords"] == ["chef"]


def test_show_professionals_instead(chat):
    say(chat, "Find chef jobs in Dubai")
    r = say(chat, "Show professionals instead")
    assert r["state"]["entity"] == "professional"
    assert r["state"]["keywords"] == ["chef"]
    assert {x["entity_type"] for x in r["results"]} == {"professional"}


def test_weak_noun_never_switches_entity(chat):
    say(chat, "Find chef jobs in Dubai")
    say(chat, "Show professionals instead")
    r = say(chat, "Only management positions")
    assert r["state"]["entity"] == "professional"


# ---------------------------------------------------------------------------
# References and details
# ---------------------------------------------------------------------------


def test_tell_me_more_about_the_first_one(chat):
    r1 = say(chat, "Find chef jobs in Dubai")
    r = say(chat, "Tell me more about the first one")
    assert r["action"] == "detail"
    assert job_ids(r) == job_ids(r1)[:1]
    assert r["detail"]["title"] == r1["results"][0]["title"]
    assert r["results"][0]["url"] == r1["results"][0]["url"]


def test_open_the_second_one(chat, url_templates):
    r1 = say(chat, "Find chef jobs in Dubai")
    r = say(chat, "Open the second one")
    assert r["action"] == "detail"
    assert job_ids(r) == job_ids(r1)[1:2]
    assert r["results"][0]["url"] == r1["results"][1]["url"]
    assert r["results"][0]["url"].startswith("https://www.hozpitality.com/test-jobs/")


def test_open_without_url_says_so(chat):
    say(chat, "Find chef jobs in Dubai")
    r = say(chat, "Open the second one")
    assert r["results"][0]["url"] is None
    assert "doesn't have a link" in r["answer"]


def test_what_company_is_the_first_job_from(chat):
    r1 = say(chat, "Find chef jobs in Dubai")
    r = say(chat, "What company is the first job from?")
    assert (
        r["answer"].endswith(f"is from {r1['results'][0]['company']}.")
        or r1["results"][0]["company"] in r["answer"]
    )


def test_reference_beyond_list_is_refused_not_guessed(chat):
    say(chat, "Find chef jobs in Abu Dhabi")  # 2 results
    r = say(chat, "Tell me more about the fifth one")
    assert r["action"] == "detail"
    assert r["results"] == []
    assert "don't have a fifth result" in r["answer"]


def test_that_company_and_previous_job(chat):
    r1 = say(chat, "Find chef jobs in Dubai")
    say(chat, "Tell me more about the second one")
    r = say(chat, "Tell me about that company")
    assert r1["results"][1]["company"] in r["answer"]
    say(chat, "Tell me more about the third one")
    r_prev = say(chat, "Tell me about the previous job")
    assert job_ids(r_prev) == job_ids(r1)[1:2]


def test_companies_hiring_these_chefs(chat):
    r1 = say(chat, "Find chef jobs in Dubai")
    r = say(chat, "What companies are hiring these chefs?")
    assert r["action"] == "related_entity"
    assert {x["entity_type"] for x in r["results"]} <= {"company"}
    companies = {x["company"] for x in r1["results"] if x.get("company")}
    assert {x["title"] for x in r["results"]} <= companies
    assert r["state"]["entity"] == "job"  # the search itself is unchanged


# ---------------------------------------------------------------------------
# Show more / compare edge cases
# ---------------------------------------------------------------------------


def test_show_more_until_exhausted_then_related(chat):
    say(chat, "Find chef jobs in Dubai")
    seen: set[str] = set()
    for _ in range(4):
        r = say(chat, "Show me more")
        ids = set(job_ids(r))
        assert not ids & seen
        seen |= ids
        if not r["results"]:
            break
    assert r["results"] == []
    assert "no more" in r["answer"].casefold()
    assert not set(job_ids(r, "related_results")) & seen


def test_show_more_without_search(chat):
    r = say(chat, "Show me more")
    assert r["action"] == "more"
    assert "Run a search first" in r["answer"]


def test_compare_without_results(chat):
    r = say(chat, "Compare the first three")
    assert r["action"] == "compare"
    assert (
        r["answer"]
        == "I don't have three recent results to compare yet. Run a search first."
    )
    assert r["results"] == []


def test_compare_more_than_available(chat):
    say(chat, "Find chef jobs in Abu Dhabi")  # 2 results
    r = say(chat, "Compare the first three")
    assert len(r["comparison"]["items"]) == 2
    assert "only has 2 results" in r["answer"]


def test_which_one_has_more_experience(chat):
    say(chat, "Find chef jobs in Dubai")
    say(chat, "Only management positions")
    r = say(chat, "Which one has more experience?")
    assert r["action"] == "compare"
    assert [f["key"] for f in r["comparison"]["fields"]] == ["experience"]
    assert "mentions the most experience" in r["answer"]


def test_compare_salary_and_location_uses_only_real_values(chat):
    say(chat, "Find executive chef jobs in Dubai")
    r = say(chat, "Compare salary and location")
    keys = [f["key"] for f in r["comparison"]["fields"]]
    assert keys == ["location", "salary"]
    salaries = {
        item["title"]: item["values"]["salary"] for item in r["comparison"]["items"]
    }
    # job.salary.description exactly as migrated
    assert salaries["Executive Chef"] == "AED 15,000 - 18,000 per month"
    assert all(
        v == "Not specified" for t, v in salaries.items() if t != "Executive Chef"
    )


# ---------------------------------------------------------------------------
# Clarification and state lifecycle
# ---------------------------------------------------------------------------


def test_clarification_flow_type_then_location(chat):
    r1 = say(chat, "Find me a job")
    assert (
        r1["action"] == "clarify"
        and r1["answer"] == "What type of job are you looking for?"
    )
    r2 = say(chat, "chef")
    assert r2["state"]["entity"] == "job"  # "chef" did not become a professional search
    assert r2["answer"] == "Which location would you prefer?"
    r3 = say(chat, "Dubai")
    assert r3["action"] == "search"
    assert {x["entity_type"] for x in r3["results"]} == {"job"}


def test_unknown_location_reply_is_explicit_location(chat):
    say(chat, "Find chef jobs")
    r = say(chat, "Antarctica")
    assert r["state"]["location"].get("raw") == "Antarctica"
    assert r["results"] == []
    assert r["related_results"]


def test_start_over_variants(chat):
    for phrase in ("Start over", "New search", "Clear", "Reset"):
        say(chat, "Find chef jobs in Dubai")
        r = say(chat, phrase)
        assert r["action"] == "reset", phrase
        assert r["state"]["entity"] is None


def test_state_is_persisted_in_mongodb(container, mongo_db):
    say(container.chat, "Find chef jobs in Dubai")
    say(container.chat, "Only management positions")
    stored = mongo_db["ai_search_conversations"].find_one({"conversation_id": CID})
    assert stored["state"]["entity"] == "job"
    assert stored["state"]["filters"] == {"level": "manager"}
    assert stored["version"] == 2
    assert stored["expires_at"] > stored["updated_at"]
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"] * 2

    # A brand-new service instance (e.g. another worker) continues the conversation.
    fresh = ChatService(container.search, container.conversations, container.llm)
    r = say(fresh, "With accommodation")
    assert r["state"]["filters"] == {"level": "manager", "accommodation": True}


def test_history_and_messages_are_bounded(container, mongo_db):
    for _ in range(3):
        say(container.chat, "Find chef jobs in Dubai")
        for _ in range(3):
            say(container.chat, "Show me more")
    stored = mongo_db["ai_search_conversations"].find_one({"conversation_id": CID})
    assert len(stored["messages"]) <= 20
    assert len(stored["state"]["result_history"]) <= 50
    assert len(stored["state"]["shown"]) <= 100


def test_legacy_prototype_state_is_migrated():
    legacy = {
        "entity": "job",
        "location": {"city": "Dubai", "country": "United Arab Emirates"},
        "keywords": ["chef"],
        "filters": {"level": "manager"},
        "last_results": ["job:1"],
        "last_result_items": [
            {"entity_type": "job", "entity_id": "1", "title": "Chef", "location": {}}
        ],
    }
    state = normalize_state(legacy)
    assert state["result_history"][0]["key"] == "job:1"
    assert state["current_list"] == ["job:1"]
    assert state["topic"]


def test_client_supplied_conversation_id_is_kept(chat):
    r = say(chat, "Find chef jobs in Dubai", cid="conv-1727300000-abc1234")
    assert r["conversation_id"] == "conv-1727300000-abc1234"
    r2 = say(chat, "Only management positions", cid="conv-1727300000-abc1234")
    assert r2["state"]["filters"]["level"] == "manager"


def test_concurrent_write_is_detected(container):
    from ai_search.app.conversation import ConversationConflict

    chat = container.chat
    turn_a = chat.prepare(message="Find chef jobs in Dubai", conversation_id=CID)
    turn_b = chat.prepare(message="Find chef jobs in Mumbai", conversation_id=CID)
    chat.finalize(turn_a, None)
    with pytest.raises(ConversationConflict):
        chat.finalize(turn_b, None)


def test_owner_binding(container):
    from ai_search.app.conversation import ConversationForbidden

    container.chat.chat(
        message="Find chef jobs in Dubai", conversation_id=CID, owner_id="user-1"
    )
    with pytest.raises(ConversationForbidden):
        container.chat.chat(
            message="Show me more", conversation_id=CID, owner_id="user-2"
        )
