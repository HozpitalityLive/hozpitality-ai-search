from ai_v4.engine.execution.search_execution import SearchExecution
from ai_v4.engine.execution.chat_execution import ChatExecution
from ai_v4.engine.execution.clarification_execution import ClarificationExecution
class ExecutionManager:
    def __init__(self): self.search=SearchExecution();self.chat=ChatExecution();self.clarification=ClarificationExecution()
    async def execute(self,user_id,websocket,query,plan,memory):
        t=plan["execution"]["type"]
        if t=="chat": return await self.chat.execute(websocket,query,plan,memory)
        if t=="clarification": return await self.clarification.execute(user_id,websocket,query,plan,memory)
        return await self.search.execute(user_id,websocket,query,plan,memory)
