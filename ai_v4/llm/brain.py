from ai_v4.llm.prompts.manager import PromptManager
from ai_v4.llm.ollama import OllamaClient
class Brain:
    def __init__(self): self.prompt_manager=PromptManager();self.ollama=OllamaClient()
    async def think(self,agent,query,context,memory=None,model=None):
        prompt=self.prompt_manager.build(agent,query,context,memory)
        async for chunk in self.ollama.stream(prompt,model): yield chunk
