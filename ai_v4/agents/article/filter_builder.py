class ArticleFilterBuilder:

    def build(
        self,
        plan: dict
    ):

        entities = plan.get(
            "entities",
            {}
        )

        filters = {
            "category": "article"
        }

        if entities.get("locations"):
            filters["location"] = entities["locations"][0]

        if entities.get("category"):
            filters["category"] = entities["category"]

        return filters