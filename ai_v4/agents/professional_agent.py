from ai_v4.agents.base import BaseAgent
from ai_v4.agents.professional.filter_builder import ProfessionalFilterBuilder


class ProfessionalAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="professional",
            builder=ProfessionalFilterBuilder()
        )