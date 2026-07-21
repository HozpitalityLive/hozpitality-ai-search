import json
import httpx

from ai_v4.config.logger import logger

from ai_v4.config.settings import settings

print(settings.OLLAMA_URL)
print(settings.DEFAULT_MODEL)


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
            "stream": True,
            "keep_alive": "24h",
            "format": "json"
        }

        logger.info("=" * 80)
        logger.info("OLLAMA REQUEST")
        logger.info(f"URL          : {url}")
        logger.info(f"MODEL        : {payload['model']}")
        logger.info(f"BASE URL     : {settings.OLLAMA_URL}")
        logger.info(f"PROMPT CHARS : {len(prompt)}")
        logger.info(f"STREAM       : {payload['stream']}")
        logger.info("=" * 80)

        try:

            async with self.client.stream(
                "POST",
                url,
                json=payload
            ) as response:

                logger.info("=" * 80)
                logger.info("OLLAMA RESPONSE")
                logger.info(f"STATUS  : {response.status_code}")
                logger.info(f"HEADERS : {dict(response.headers)}")
                logger.info("=" * 80)

                if response.status_code != 200:

                    body = await response.aread()

                    logger.error("OLLAMA ERROR")
                    logger.error(body.decode(errors="ignore"))

                    return

                async for line in response.aiter_lines():

                    if not line:
                        continue

                    logger.info(f"TOKEN: {line[:150]}")

                    yield json.loads(line)

        except Exception as e:

            logger.exception("OLLAMA EXCEPTION")
            raise

    
    async def generate(
        self,
        prompt: str,
        model: str | None = None
    ):

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model or settings.DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
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