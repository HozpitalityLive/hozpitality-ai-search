from ai_v4.planner.intent import IntentDetector
from ai_v4.planner.entities import EntityExtractor
from ai_v4.planner.router import PlannerRouter
from ai_v4.config.logger import logger
from ai_v4.planner.clarification import ClarificationDetector
from ai_v4.planner.intent import Intent


class Planner:

    def __init__(self):
        self.intent = IntentDetector()
        self.entities = EntityExtractor()
        self.router = PlannerRouter()
        self.clarification = ClarificationDetector()

    async def create_plan(
        self,
        query: str
    ):
        
        logger.info("[1/4] Creating execution plan...")

        intent = await self.intent.detect(query)
        entities = await self.entities.extract(query)
        route = await self.router.route(
            intent=intent,
            entities=entities
        )

        execution = {
            "type": "search"
        }

        if intent == Intent.GREETING:
            execution["type"] = "chat"

        plan = {
            "query": query,
            "intent": intent,
            "entities": entities,
            "route": route,
            "search": {
                "engines": [
                    "elastic",
                    "postgres"
                ],
                "limit": 20,
                "rerank": True
            },
            "execution": execution,

            "clarification": {
                "required": False,
                "question": None,
                "missing": []
            },

            "llm": {
                "model": "llama3-hoz:latest",
                "temperature": 0.2
            }
        }

        return plan