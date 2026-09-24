from ai_search.app.ranking import score_document


def test_exact_title_wins_over_description():
    exact, exact_matches = score_document(
        {"title": "Executive Chef", "keywords": ["chef"], "description": ""},
        "executive chef",
    )
    weak, weak_matches = score_document(
        {"title": "Hotel Manager", "keywords": [], "description": "Executive chef support"},
        "executive chef",
    )
    assert exact > weak
    assert "exact_title" in exact_matches
