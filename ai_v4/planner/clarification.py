class ClarificationDetector:

    async def analyze(
        self,
        query,
        intent,
        entities,
    ):

        return {
            "required": False,
            "question": None,
            "missing": []
        }