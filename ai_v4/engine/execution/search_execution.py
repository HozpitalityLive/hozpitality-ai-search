import json
from ai_v4.services.agent_service import AgentService
from ai_v4.services.search_service import SearchService
from ai_v4.context.builder import ContextBuilder
from ai_v4.llm.response import ResponseGenerator
from ai_v4.services.memory_service import MemoryService

class SearchExecution:
    def __init__(self):
        self.agent_service=AgentService();self.search_service=SearchService();self.context_builder=ContextBuilder()
        self.response=ResponseGenerator();self.memory_service=MemoryService()
    async def execute(self,user_id,websocket,query,plan,memory):
        # Analytics gets a direct SQL count/aggregation path.
        if plan["intent"]=="job_analytics":
            analytics=await self.search_service.analytics(query,plan.get("filters",{}))
            if analytics:
                results=[{"engine":"postgres","score":1,"document":x} for x in analytics]
                agent_output={"agent":"job","filters":plan.get("filters",{}),"results":results,"total":len(results)}
            else:
                agent_output=await self.agent_service.execute(plan,query,memory)
        else:
            agent_output=await self.agent_service.execute(plan,query,memory)
        memory["last_search"]={"query":query,"plan":plan,"filters":plan.get("filters",{}),"agent":agent_output.get("agent"),"results":agent_output.get("results",[])[:20]}
        context=await self.context_builder.build(query,agent_output.get("results",[]),memory)
        response=await self.response.generate(websocket,agent_output.get("agent","search"),query,context,memory,plan["llm"]["model"])
        memory["conversation"] += [{"role":"user","content":query},{"role":"assistant","content":response.get("intro","")}]
        await self.memory_service.save(user_id,memory)
        return response
