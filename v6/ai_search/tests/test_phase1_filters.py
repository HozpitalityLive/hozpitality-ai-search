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


def test_job_city_filter_does_not_use_company_or_author_city():
    query = SearchDocumentsRepository._filter(
        entity="job", city="Dubai", country=None, status=None, is_live=None
    )
    text = repr(query)
    assert "company.city" not in text
    assert "author.city_town" not in text


def test_job_location_is_authoritative():
    doc = {
        "entity_type": "job",
        "location": {"city": "Koh Krabey Island", "country": {"name": "Cambodia"}},
        "company": {"city": "Dubai", "country": {"name": "United Arab Emirates"}},
    }
    assert not SearchDocumentsRepository._document_matches_filters(
        doc, entity="job", city="Dubai", country="United Arab Emirates",
        status=None, is_live=None
    )
