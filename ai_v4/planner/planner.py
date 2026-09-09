from ai_v4.planner.intent import IntentDetector,Intent
from ai_v4.planner.extractors.semantic_extractor import SemanticExtractor
from ai_v4.planner.query_parser import QueryParser
from ai_v4.planner.search_planner import SearchPlanner
from ai_v4.planner.router import PlannerRouter
from ai_v4.planner.clarification import ClarificationDetector

class Planner:
    def __init__(self):
        self.intent=IntentDetector(); self.semantic=SemanticExtractor(); self.parser=QueryParser()
        self.search_planner=SearchPlanner(); self.router=PlannerRouter(); self.clarification=ClarificationDetector()
    async def create_plan(self,query):
        intent=await self.intent.detect(query)
        entities=await self.semantic.extract(query)
        filters=await self.parser.parse(query,intent,entities)
        route=await self.router.route(intent,filters)
        sp=await self.search_planner.build(intent,filters)
        clarification=await self.clarification.analyze(query,intent,sp)
        execution="chat" if intent in (Intent.GREETING,Intent.CHAT) else ("clarification" if clarification["required"] else "search")
        return {"query":query,"intent":intent.value,"entities":entities,"filters":filters,"search_plan":sp,
                "route":route,"execution":{"type":execution},"clarification":clarification,
                "search":{"engines":["elastic","postgres","vector"],"limit":20,"rerank":True},
                "llm":{"model":"llama3-hoz:latest","temperature":0.2}}
