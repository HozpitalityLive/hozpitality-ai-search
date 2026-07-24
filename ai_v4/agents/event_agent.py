from ai_v4.agents.base import BaseAgent
from ai_v4.agents.event.filter_builder import EventFilterBuilder


class EventAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="event",
            builder=EventFilterBuilder()
        )