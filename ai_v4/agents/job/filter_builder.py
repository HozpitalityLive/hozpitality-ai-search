class JobFilterBuilder:

    def build(
        self,
        plan: dict
    ):

        entities = plan.get(
            "entities",
            {}
        )

        filters = {
            "category": "job"
        }

        if entities.get("locations"):
            filters["location"] = entities["locations"][0]

        if entities.get("skills"):
            filters["skills"] = entities["skills"]

        if entities.get("roles"):
            filters["role"] = entities["roles"][0]

        if entities.get("experience"):
            filters["experience"] = entities["experience"]

        return filters