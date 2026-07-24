class AwardsFilterBuilder:

    def build(
        self,
        plan: dict
    ):

        entities = plan.get(
            "entities",
            {}
        )

        filters = {
            "category": "awards"
        }

        # if entities.get("locations"):
        #     filters["location"] = entities["locations"][0]

        # if entities.get("industry"):
        #     filters["industry"] = entities["skills"]

        return filters