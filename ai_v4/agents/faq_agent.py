from ai_v4.agents.base import BaseAgent
class FaqAgent(BaseAgent):
    def __init__(self): super().__init__("faq",None)
    async def execute(self,query,plan,memory):
        filters={"category":"faq"}
        results=await self.search.search(query,filters,10)
        return {"agent":"faq","query":query,"filters":filters,"total":len(results),"page":1,"page_size":5,"results":results}
