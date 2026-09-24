from __future__ import annotations

from ai_search.app.chat_service import ChatService
from ai_search.app.query_understanding import understand


def test_accommodation_filter_is_understood():
    plan = understand("chef jobs in Dubai with accommodation")
    assert plan.entity == "job"
    assert plan.city == "Dubai"
    assert plan.filters["accommodation"] is True
    assert "accommodation" not in plan.keywords


def test_follow_up_state_preserves_search_context():
    first = understand("chef jobs in Dubai")
    state = ChatService._state_from_plan(first, {})

    follow_up = understand("only management positions")
    state = ChatService._state_from_plan(follow_up, state)

    assert state["entity"] == "job"
    assert state["keywords"] == ["chef"]
    assert state["location"]["city"] == "Dubai"
    assert state["filters"]["level"] == "manager"
    assert ChatService._state_query(state) == "chef manager"


def test_accommodation_follow_up_merges_with_existing_state():
    first = understand("chef jobs in Dubai")
    state = ChatService._state_from_plan(first, {})

    follow_up = understand("with accommodation")
    state = ChatService._state_from_plan(follow_up, state)

    assert state["entity"] == "job"
    assert state["keywords"] == ["chef"]
    assert state["location"]["city"] == "Dubai"
    assert state["filters"]["accommodation"] is True
    assert "accommodation" in ChatService._state_query(state)


def test_chat_action_detection():
    assert ChatService._action("Show me more") == "more"
    assert ChatService._action("Compare the first three") == "compare"
    assert ChatService._action("Start over") == "reset"
    assert ChatService._action("Only management positions") == "search"
