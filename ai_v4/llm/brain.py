from ai_v4.llm.prompts.manager import PromptManager
from ai_v4.llm.ollama import OllamaClient
from ai_v4.config.logger import logger


class Brain:

    def __init__(self):
        self.prompt_manager = PromptManager()
        self.ollama = OllamaClient()

    async def think(
        self,
        agent: str,
        query: str,
        context: dict,
        memory: dict | None = None,
        model=None
    ):

        logger.info(f"Brain Thinking ({agent})")

        logger.info("=" * 80)
        logger.info("CONTEXT PASSED TO BRAIN")
        logger.info(context)
        logger.info("=" * 80)

        prompt = self.prompt_manager.build(
            agent=agent,
            query=query,
            context=context,
            memory=memory
        )

        logger.info("=" * 80)
        logger.info("PROMPT")
        logger.info(prompt)
        logger.info("=" * 80)

        async for chunk in self.ollama.stream(
            prompt=prompt,
            model=model
        ):
            yield chunk