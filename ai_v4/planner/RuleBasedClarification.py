from ai_v4.planner.intent import Intent



class RuleBasedClarification:
    def get_clarifications(
        self,
        intent,
        entities
    ):
        if intent == Intent.PROFESSIONAL_SEARCH:

            if entities.get("person_names"):
                return {
                    "required": False,
                    "question": None,
                    "missing": [],
                    "reason": "Specific person detected."
                }

        if intent == Intent.COMPANY_SEARCH:

            if entities.get("company_names"):
                return {
                    "required": False,
                    "question": None,
                    "missing": [],
                    "reason": "Specific company detected."
                }

        return None