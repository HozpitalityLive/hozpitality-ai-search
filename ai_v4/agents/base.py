from abc import ABC, abstractmethod

from ai_v4.services.search_service import SearchService


class BaseAgent(ABC):

    def __init__(
        self,
        name: str

    ):
        self.name = name
        self.search_service = SearchService()

    @abstractmethod
    async def execute(
        self,
        query: str,
        plan: dict,
        memory: dict
    ):
        pass