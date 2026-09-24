from ai_search.app.ranking import score_document


def test_exact_title_wins_over_description():
    exact, exact_matches = score_document(
        {"title": "Executive Chef", "search_keywords": ["chef"], "ai_search_text": ""},
        "executive chef",
    )
    weak, weak_matches = score_document(
        {"title": "Hotel Manager", "search_keywords": [], "ai_search_text": "Executive chef support"},
        "executive chef",
    )
    assert exact > weak
    assert "exact_title" in exact_matches
