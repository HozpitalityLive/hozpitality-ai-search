from fastapi import WebSocket
from ai_v4.engine.execution.manager import ExecutionManager
from ai_v4.planner.planner import Planner
from ai_v4.planner.query_rewriter import QueryRewriter
from ai_v4.services.memory_service import MemoryService

class AIEngine:
    def __init__(self):
        self.planner=Planner();self.memory_service=MemoryService();self.query_rewriter=QueryRewriter();self.execution=ExecutionManager()

    async def execute(self,user_id,websocket,query,memory=None):
        memory=await self.memory_service.load(user_id)

        # A clarification is a pending first-turn query.
        pending=memory.get("pending_query")
        if pending:
            query=await self.query_rewriter.rewrite(pending,query)
            memory["pending_query"]=None
        else:
            # Natural follow-up modifiers inherit the previous search.
            last=memory.get("last_search") or {}
            low=query.lower().strip()
            modifiers=("only ","show ","filter ","sort ","highest ","lowest ","more ","less ","exclude ","not ")
            if last.get("query") and (low.startswith(modifiers) or low in {"senior","junior","latest","more"}):
                query=await self.query_rewriter.rewrite(last["query"],query)

        plan=await self.planner.create_plan(query)
        result=await self.execution.execute(user_id,websocket,query,plan,memory)
        if plan["execution"]["type"]=="clarification":
            return result
        return result
