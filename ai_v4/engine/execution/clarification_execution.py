class ClarificationExecution:
    async def execute(self,user_id,websocket,query,plan,memory):
        q=plan["clarification"]["question"]
        await websocket.send_json({"type":"clarification","question":q})
        memory["pending_query"]=query
        from ai_v4.services.memory_service import MemoryService
        await MemoryService().save(user_id,memory)
        return None
