import json
import httpx

from ai_v4.config.logger import logger
from ai_v4.config.settings import settings


class OllamaClient:

    def __init__(self):
        self.base_url = settings.OLLAMA_URL
        self.model = settings.DEFAULT_MODEL
        self.client = httpx.AsyncClient(
            timeout=None
        )

    async def stream(
        self,
        prompt: str,
        model: str | None = None
    ):

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model or settings.DEFAULT_MODEL,
            "prompt": prompt,
            "stream": True
        }

        async with self.client.stream(
            "POST",
            url,
            json=payload
        ) as response:

            async for line in response.aiter_lines():
                if not line:
                    continue
                yield json.loads(line)

    
    async def generate(
        self,
        prompt: str,
        model: str | None = None
    ):

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model or settings.DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False
        }

        response = await self.client.post(
            url,
            json=payload
        )

        return response.json()
    

    async def health(self):
        try:
            response = await self.client.get(
                f"{self.base_url}/api/tags"
            )

            return response.status_code == 200
        except:
            return False
        
    
    async def models(self):

        response = await self.client.get(

            f"{self.base_url}/api/tags"

        )

        return response.json()