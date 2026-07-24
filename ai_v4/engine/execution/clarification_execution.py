from ai_v4.services.memory_service import MemoryService


class ClarificationExecution:

    def __init__(self):
        self.memory_service = MemoryService()

    async def execute(
        self,
        user_id,
        websocket,
        query,
        plan,
        memory
    ):

        await websocket.send_json({
            "type": "clarification",
            "question": plan["clarification"]["question"]
        })

        memory["pending_query"] = query

        await self.memory_service.save(
            user_id,
            memory
        )

        return None