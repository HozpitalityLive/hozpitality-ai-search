from ai_search.app.query_understanding import clarification_for, understand


def test_natural_language_job_query():
    plan = understand("I need a senior chef job in Dubai with 5 years experience")
    assert plan.intent == "search"
    assert plan.entity == "job"
    assert "chef" in plan.keywords
    assert plan.city == "Dubai"
    assert plan.country == "United Arab Emirates"
    assert plan.experience == 5
    assert plan.level == "senior"


def test_clarification_job_without_type():
    plan = understand("Find me a job")
    assert clarification_for(plan) == "What type of job are you looking for?"


def test_clarification_job_without_location():
    plan = understand("Find chef jobs")
    assert clarification_for(plan) == "Which location would you prefer?"


def test_direct_search_when_location_is_present():
    plan = understand("Find chef jobs in Dubai")
    assert clarification_for(plan) is None


def test_all_entity_clarifications():
    cases = {
        "Find professionals": "What type of professional or skill are you looking for?",
        "Find companies": "What type of company or hospitality business are you looking for?",
        "Find products": "What type of product or supplier are you looking for?",
        "Find articles": "What topic or category would you like me to search for in the articles?",
        "Find events": "What type of event are you looking for?",
        "Find awards": "What type of award are you looking for?",
        "Find FAQs": "What topic or question should I search for?",
    }
    for query, expected in cases.items():
        assert clarification_for(understand(query)) == expected


def test_professional_role_query_is_not_a_job():
    plan = understand("find senior chefs in Dubai")
    assert plan.entity == "professional"
    assert plan.keywords == ["chefs"]
    assert plan.city == "Dubai"
    assert plan.country == "United Arab Emirates"
    assert plan.level == "senior"


def test_no_unrequested_filters():
    plan = understand("Find me a job")
    assert plan.filters == {}


def test_hotel_companies_keeps_hotel_as_keyword():
    plan = understand("hotel companies in Dubai")
    assert plan.entity == "company"
    assert "hotel" in plan.keywords
    assert plan.city == "Dubai"
    assert plan.country == "United Arab Emirates"
    assert clarification_for(plan) is None


def test_long_natural_language_preserves_executive_chef_role():
    plan = understand(
        "I am looking for an experienced executive chef who can manage "
        "a luxury hotel kitchen in Dubai"
    )
    assert plan.entity == "professional"
    assert plan.city == "Dubai"
    assert plan.country == "United Arab Emirates"
    assert "executive" in plan.keywords
    assert "chef" in plan.keywords

def test_long_natural_language_removes_conversational_filler():
    plan = understand(
        "I am looking for an experienced executive chef who can manage "
        "a luxury hotel kitchen in Dubai"
    )
    assert "am" not in plan.keywords
    assert "experienced" not in plan.keywords
    assert "who" not in plan.keywords
    assert "executive" in plan.keywords
    assert "chef" in plan.keywords
