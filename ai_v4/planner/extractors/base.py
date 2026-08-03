from abc import ABC, abstractmethod


class BaseExtractor(ABC):

    @abstractmethod
    async def extract(
        self,
        query: str
    ) -> dict:
        pass