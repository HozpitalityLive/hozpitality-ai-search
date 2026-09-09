from abc import ABC
from ai_v4.services.search_service import SearchService
class BaseAgent(ABC):
    def __init__(self,name,builder=None):
        self.name=name;self.builder=builder;self.search=SearchService()
    async def execute(self,query,plan,memory):
        filters=self.builder.build(plan) if self.builder else {}
        results=await self.search.search(query,filters,plan.get("search",{}).get("limit",20))
        return {"agent":self.name,"query":query,"filters":filters,"total":len(results),"page":1,"page_size":5,"results":results}
