from ai_v4.llm.response import ResponseGenerator
class ChatExecution:
    async def execute(self,websocket,query,plan,memory):
        return await ResponseGenerator().generate(websocket,"chat",query,{"query":query,"memory":memory,"documents":"","results":[],"total":0},memory,plan["llm"]["model"])
