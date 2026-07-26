from abc import ABC, abstractmethod


class BaseParser(ABC):

    @abstractmethod
    async def parse(
        self,
        query: str,
        entities: dict,
        filters: dict,
    ) -> dict:
        pass