class CompanyFilterBuilder:

    def build(
        self,
        plan: dict
    ):

        entities = plan.get(
            "entities",
            {}
        )

        filters = {
            "category": "company"
        }

        if entities.get("locations"):
            filters["location"] = entities["locations"][0]

        if entities.get("industry"):
            filters["industry"] = entities["industry"]

        return filters