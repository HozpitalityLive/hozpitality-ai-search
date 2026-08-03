from ai_v4.planner.intent import IntentDetector
from ai_v4.planner.extractors.semantic_extractor import SemanticExtractor
from ai_v4.planner.query_parser import QueryParser
from ai_v4.planner.search_planner import SearchPlanner
from ai_v4.planner.router import PlannerRouter
from ai_v4.config.logger import logger
from ai_v4.planner.clarification import ClarificationDetector
from ai_v4.planner.intent import Intent


class Planner:

    def __init__(self):
        self.intent = IntentDetector()
        self.semantic = SemanticExtractor()
        self.router = PlannerRouter()
        self.clarification = ClarificationDetector()
        self.parser = QueryParser()
        self.search_planner = SearchPlanner()

    async def create_plan(
        self,
        query: str
    ):
        
        logger.info("[1/4] Creating execution plan...")

        intent = await self.intent.detect(query)
        # entities = await self.entities.extract(query)
        entities = await self.semantic.extract(query)
        logger.info("="*60)
        logger.info("ENTITIES")
        logger.info(entities)
        logger.info("="*60)
        
        filters = await self.parser.parse(
            query=query,
            intent=intent,
            entities=entities
        )

        route = await self.router.route(
            intent=intent,
            entities=filters
        )

        search_plan = await self.search_planner.build(
            intent=intent,
            filters=filters
        )

        if search_plan["can_search"]:
            clarification = {
                "required": False,
                "question": None,
                "missing": []
            }
        else:
            clarification = await self.clarification.analyze(
                query=query,
                intent=intent,
                search_plan=search_plan
            )

        execution = {
            "type": "search"
        }

        if intent in (
            Intent.GREETING,
            Intent.CHAT
        ):
            execution["type"] = "chat"

        elif clarification["required"]:
            execution["type"] = "clarification"

        plan = {
            "query": query,
            "intent": intent,
            "entities": entities,
            "filters": filters,
            "search_plan": search_plan,
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

            "clarification": clarification,

            "llm": {
                "model": "llama3-hoz:latest",
                "temperature": 0.2
            }
        }

        return plan