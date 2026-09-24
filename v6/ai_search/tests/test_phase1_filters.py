from ai_search.app.repository import SearchDocumentsRepository


def test_filter_city_is_hard_constraint():
    query = SearchDocumentsRepository._filter(
        entity="job",
        city="Dubai",
        country=None,
        status=None,
        is_live=None,
    )

    assert "$and" in query
    assert {"entity_type": "job"} in query["$and"]
    assert any("$or" in clause for clause in query["$and"])


def test_country_alias_expands_uae():
    assert SearchDocumentsRepository._country_terms("UAE") == [
        "United Arab Emirates",
        "AE",
        "UAE",
    ]
