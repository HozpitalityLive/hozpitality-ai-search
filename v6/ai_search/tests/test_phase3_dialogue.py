"""Deterministic turn interpretation (no MongoDB, no LLM)."""

from __future__ import annotations

import pytest

from ai_search.app.dialogue import interpret

JOB_STATE = {
    "entity": "job",
    "keywords": ["chef"],
    "location": {"city": "Dubai", "country": "United Arab Emirates"},
    "filters": {},
    "last_results": ["job:1", "job:2", "job:3"],
}


@pytest.mark.parametrize(
    "message",
    [
        "Show me more",
        "More",
        "Show additional results",
        "Anything else?",
        "more results",
        "next page",
        "show me more jobs",
        "what else",
    ],
)
def test_show_more_variants(message):
    assert interpret(message, JOB_STATE).action == "more"


@pytest.mark.parametrize("message", ["more senior", "more senior roles"])
def test_more_as_refinement_is_not_pagination(message):
    intent = interpret(message, JOB_STATE)
    assert intent.action == "search"
    assert intent.set_filters.get("level") == "senior"


@pytest.mark.parametrize(
    "message,refs",
    [
        ("Compare the first three", [0, 1, 2]),
        ("Compare the first two", [0, 1]),
        ("compare 1 and 3", [0, 2]),
        ("compare the second and third", [1, 2]),
    ],
)
def test_compare_references(message, refs):
    intent = interpret(message, JOB_STATE)
    assert intent.action == "compare"
    assert intent.refs == refs


def test_compare_these_and_fields():
    assert interpret("Compare these", JOB_STATE).ref_mode == "all"
    assert interpret("Which one has more experience?", JOB_STATE).compare_fields == [
        "experience"
    ]
    assert interpret("Compare salary and location", JOB_STATE).compare_fields == [
        "salary",
        "location",
    ]


@pytest.mark.parametrize(
    "message,ref,field",
    [
        ("Tell me more about the first one", 0, None),
        ("Open the second job", 1, "url"),
        ("Open the second one", 1, "url"),
        ("What company is the first job from?", 0, "company"),
        ("Show me details of the third one", 2, None),
        ("the second one", 1, None),
        ("what is the salary of #2", 1, "salary"),
    ],
)
def test_result_references(message, ref, field):
    intent = interpret(message, JOB_STATE)
    assert intent.action == "detail"
    assert intent.refs == [ref]
    assert intent.question_field == field


def test_focus_and_previous_references():
    state = {**JOB_STATE, "focus": "job:2", "previous_focus": "job:1"}
    assert interpret("Tell me about that company", state).ref_mode == "focus"
    assert interpret("Tell me about the previous job", state).ref_mode == "previous"


@pytest.mark.parametrize(
    "message",
    ["Start over", "New search", "Clear", "Reset", "start again", "reset the search"],
)
def test_reset_variants(message):
    assert interpret(message, JOB_STATE).action == "reset"


def test_clear_filter_is_not_a_reset():
    intent = interpret("Clear the location", JOB_STATE)
    assert intent.action == "search"
    assert intent.remove_location


def test_only_management_positions_keeps_entity():
    intent = interpret("Only management positions", JOB_STATE)
    assert intent.set_entity is None
    assert intent.set_filters == {"level": "manager"}
    assert intent.strict_filters == ["level"]
    assert intent.set_keywords is None and intent.add_keywords == []


def test_role_word_never_flips_job_to_professional():
    intent = interpret("senior chef", JOB_STATE)
    assert intent.set_entity is None


def test_with_accommodation():
    intent = interpret("With accommodation", JOB_STATE)
    assert intent.set_filters == {"accommodation": True}
    assert "accommodation" in intent.strict_filters


@pytest.mark.parametrize(
    "message,removed",
    [
        ("Remove the accommodation requirement", ["accommodation"]),
        ("I don't need accommodation", ["accommodation"]),
        ("Don't restrict it to management", ["level"]),
        ("any level", ["level"]),
        ("remove the salary filter", ["salary_min", "salary_currency"]),
    ],
)
def test_filter_removal(message, removed):
    intent = interpret(
        message, {**JOB_STATE, "filters": {"accommodation": True, "level": "manager"}}
    )
    assert intent.remove_filters == removed
    assert not any(key in intent.set_filters for key in removed)


def test_show_all_management_positions_adds_not_removes():
    intent = interpret("Show all management positions", JOB_STATE)
    assert intent.remove_filters == []
    assert intent.set_filters["level"] == "manager"


@pytest.mark.parametrize(
    "message,city,country",
    [
        ("Actually Abu Dhabi", "Abu Dhabi", "United Arab Emirates"),
        ("Actually, show me Abu Dhabi", "Abu Dhabi", "United Arab Emirates"),
        ("Change that to Mumbai", "Mumbai", "India"),
        ("what about Doha?", "Doha", "Qatar"),
    ],
)
def test_location_changes(message, city, country):
    intent = interpret(message, JOB_STATE)
    assert intent.set_location == {"city": city, "country": country, "raw": None}
    assert intent.set_keywords is None


def test_anywhere_variants():
    assert interpret("Anywhere, not just Dubai", JOB_STATE).remove_location
    assert interpret("Remove the location", JOB_STATE).remove_location
    uae = interpret("Anywhere in UAE", JOB_STATE)
    assert uae.remove_location and uae.set_location["country"] == "United Arab Emirates"


def test_explicit_entity_changes():
    assert (
        interpret("Show me professionals instead", JOB_STATE).set_entity
        == "professional"
    )
    assert (
        interpret("Actually show professionals instead", JOB_STATE).set_entity
        == "professional"
    )
    assert interpret("what about companies?", JOB_STATE).set_entity == "company"


def test_related_companies_question():
    intent = interpret("What companies are hiring these chefs?", JOB_STATE)
    assert intent.action == "related_entity"
    assert intent.target_entity == "company"


def test_new_topic_replaces_keywords_and_refinement_adds():
    assert interpret("What about waiters?", JOB_STATE).set_keywords == ["waiter"]
    assert interpret("only pastry", JOB_STATE).add_keywords == ["pastry"]
    fresh = interpret("Find sous chef jobs in Mumbai", JOB_STATE)
    assert fresh.fresh and fresh.set_keywords == ["sous", "chef"]


def test_unknown_explicit_location_is_not_a_keyword():
    intent = interpret("chef jobs in Antarctica", {})
    assert intent.set_location == {"city": None, "country": None, "raw": "Antarctica"}
    assert intent.set_keywords == ["chef"]


def test_smalltalk_and_empty():
    assert interpret("hello", {}).action == "smalltalk"
    assert interpret("thanks", JOB_STATE).smalltalk == "thanks"
    assert interpret("   ", {}).action == "clarify"


def test_reference_without_results_is_not_detail():
    assert interpret("Tell me more about the first one", {}).action != "detail"
