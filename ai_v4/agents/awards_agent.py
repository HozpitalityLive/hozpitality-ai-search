from ai_v4.agents.base import BaseAgent
from ai_v4.agents.awards.filter_builder import AwardsFilterBuilder


class AwardsAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="awards",
            builder=AwardsFilterBuilder()
        )