from ai_v4.agents.base import BaseAgent
from ai_v4.agents.article.filter_builder import ArticleFilterBuilder


class ArticleAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="article",
            builder=ArticleFilterBuilder()
        )