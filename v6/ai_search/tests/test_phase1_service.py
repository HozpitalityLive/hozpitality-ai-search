from ai_search.app.service import SearchService


class FakeRepository:
    def __init__(self):
        self.docs = [
            {
                "_id": "job:1",
                "entity_type": "job",
                "source": {"object_id": 1},
                "title": "Executive Chef",
                "category": {"name": "Chef"},
                "search_keywords": ["executive chef", "chef"],
                "search_aliases": ["head chef"],
                "location": {"city": "Dubai", "country": {"name": "United Arab Emirates", "code": "AE"}},
                "ai_search_text": "Executive Chef chef head chef Dubai UAE luxury hotel",
                "text_score": 4.0,
                "is_live": True,
            },
            {
                "_id": "job:2",
                "entity_type": "job",
                "source": {"object_id": 2},
                "title": "Sous Chef",
                "category": {"name": "Chef"},
                "search_keywords": ["sous chef"],
                "search_aliases": [],
                "location": {"city": "Dubai", "country": {"name": "United Arab Emirates", "code": "AE"}},
                "ai_search_text": "Sous Chef chef Dubai UAE",
                "text_score": 3.0,
                "is_live": True,
            },
        ]

    def vocabulary(self):
        return ["executive", "chef", "sous", "head", "dubai", "uae"]

    def search(self, query, **kwargs):
        docs = list(self.docs)
        if kwargs.get("entity"):
            docs = [d for d in docs if d["entity_type"] == kwargs["entity"]]
        if kwargs.get("city"):
            docs = [d for d in docs if d["location"]["city"].casefold() == kwargs["city"].casefold()]
        return docs


def test_phase1_typo_and_top5():
    service = SearchService(FakeRepository(), fuzzy_threshold=80)
    result = service.search(query="excutive chef", entity="job", city="Dubai", limit=5)

    assert result["corrected_query"] == "executive chef"
    assert result["total"] == 2
    assert result["results"][0]["title"] == "Executive Chef"
    assert result["results"][0]["entity_id"] == "1"


def test_default_search_does_not_force_live_filter():
    class RecordingRepository(FakeRepository):
        def search(self, query, **kwargs):
            self.last_kwargs = kwargs
            return super().search(query, **kwargs)

    repo = RecordingRepository()
    service = SearchService(repo, fuzzy_threshold=80)
    service.search(query="chef", limit=5)
    assert repo.last_kwargs["is_live"] is None
