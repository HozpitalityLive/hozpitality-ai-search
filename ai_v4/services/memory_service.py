class MemoryService:

    def __init__(self):
        self._memory = {}

    async def load(self, user_id):

        return self._memory.setdefault(
            user_id,
            {
                "conversation": [],
                "last_search": {}
            }
        )

    async def save(self, user_id, memory):

        self._memory[user_id] = memory