import json,httpx
from ai_v4.config.settings import settings
class OllamaClient:
    def __init__(self): self.base_url=settings.OLLAMA_URL.rstrip("/");self.client=httpx.AsyncClient(timeout=None)
    async def stream(self,prompt,model=None):
        payload={"model":model or settings.DEFAULT_MODEL,"prompt":prompt,"stream":True,"keep_alive":"24h","format":"json"}
        async with self.client.stream("POST",f"{self.base_url}/api/generate",json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line:
                    try: yield json.loads(line)
                    except json.JSONDecodeError: continue
    async def generate(self,prompt,model=None):
        r=await self.client.post(f"{self.base_url}/api/generate",json={"model":model or settings.DEFAULT_MODEL,"prompt":prompt,"stream":False,"format":"json"})
        r.raise_for_status();return r.json()
    async def health(self):
        try:return (await self.client.get(f"{self.base_url}/api/tags")).status_code==200
        except Exception:return False
    async def models(self): return (await self.client.get(f"{self.base_url}/api/tags")).json()
