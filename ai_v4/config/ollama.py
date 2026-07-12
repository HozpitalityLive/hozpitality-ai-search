import httpx
from ai_v4.config.settings import settings


async_client = httpx.AsyncClient(
    timeout=None
)