from ai_v4.agents.base import BaseAgent
from ai_v4.agents.job.filter_builder import JobFilterBuilder


class JobAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="job",
            builder=JobFilterBuilder()
        )