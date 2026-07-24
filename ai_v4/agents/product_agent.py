from ai_v4.agents.base import BaseAgent
from ai_v4.agents.product.filter_builder import ProductFilterBuilder


class ProductAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="product",
            builder=ProductFilterBuilder()
        )