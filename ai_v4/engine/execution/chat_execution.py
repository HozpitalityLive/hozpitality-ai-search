from ai_v4.llm.response import ResponseGenerator


class ChatExecution:

    def __init__(self):
        self.response = ResponseGenerator()

    async def execute(
        self,
        websocket,
        query,
        plan,
        memory
    ):

        return await self.response.generate(
            websocket=websocket,
            agent="chat",
            query=query,
            context={
                "query": query,
                "memory": memory,
                "documents": "",
                "results": []
            },
            memory=memory,
            model=plan["llm"]["model"]
        )