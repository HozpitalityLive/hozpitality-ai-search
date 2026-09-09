from ai_v4.memory.memory_store import MemoryStore
class MemoryService:
    def __init__(self): self.store=MemoryStore()
    async def load(self,user_id): return self.store.load(user_id)
    async def save(self,user_id,memory): self.store.save(user_id,memory)
