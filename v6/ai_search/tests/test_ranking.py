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


def test_exact_match_is_not_reported_as_fuzzy():
    score, matches = score_document(
        {
            "title": "Chef",
            "search_aliases": ["chef"],
            "search_keywords": ["chef"],
            "ai_search_text": "Chef",
        },
        "chef",
    )
    assert score > 0
    assert "exact_title" in matches
    assert "fuzzy" not in matches
