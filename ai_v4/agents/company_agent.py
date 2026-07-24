from ai_v4.agents.base import BaseAgent
from ai_v4.agents.company.filter_builder import CompanyFilterBuilder


class CompanyAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="company",
            builder=CompanyFilterBuilder()
        )