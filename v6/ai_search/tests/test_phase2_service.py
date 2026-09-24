from ai_search.app.service import SearchService


class FakeRepository:
    def __init__(self):
        self.calls = []

    def suggest_vocabulary(self, token):
        return ["executive"] if token == "excutive" else [token]

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [{
            "_id": "job:1",
            "entity_type": "job",
            "title": "Executive Chef",
            "description": "Senior chef role",
            "location": {"city": "Dubai", "country": {"name": "United Arab Emirates"}},
            "source": {"object_id": 1},
            "category": {"name": "Culinary"},
            "ai_search_text": "Executive Chef senior chef Dubai",
        }]

    def fetch_by_ids(self, ids):
        return []


def test_service_extracts_and_searches_structured_query():
    repo = FakeRepository()
    service = SearchService(repo)
    result = service.search(query="I need a senior chef job in Dubai with 5 years experience")
    assert result["understanding"]["entity"] == "job"
    assert result["understanding"]["city"] == "Dubai"
    assert result["understanding"]["experience"] == 5
    assert result["understanding"]["level"] == "senior"
    assert repo.calls[0][0] == "chef"


def test_service_corrects_typo_in_keyword_only():
    repo = FakeRepository()
    service = SearchService(repo)
    result = service.search(query="excutive chef jobs in Dubai")
    assert result["corrected_query"] == "executive chef"
    assert repo.calls[0][0] == "executive chef"


def test_structured_filter_falls_back_to_soft_matching():
    class RecordingRepository(FakeRepository):
        def __init__(self):
            super().__init__()
            self.calls = []

        def search(self, query, **kwargs):
            self.calls.append((query, kwargs))
            # Simulate a schema mismatch: strict structured retrieval finds
            # nothing, while entity/location retrieval has a valid job.
            if kwargs.get("structured"):
                return []
            return super().search(query, **kwargs)

    repo = RecordingRepository()
    service = SearchService(repo, fuzzy_threshold=80)
    result = service.search(
        query="I need a senior chef job in Dubai with 5 years experience",
        limit=5,
    )
    assert result["understanding"]["entity"] == "job"
    assert result["understanding"]["city"] == "Dubai"
    assert result["understanding"]["experience"] == 5
    assert result["understanding"]["level"] == "senior"
    assert len(repo.calls) >= 2
