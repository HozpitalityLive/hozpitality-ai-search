class BasePrompt:

    def build(
        self,
        query: str,
        context: dict,
        memory: dict | None = None
    ) -> str:
        raise NotImplementedError