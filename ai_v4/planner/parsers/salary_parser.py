
import re

from .base import BaseParser


class SalaryParser(BaseParser):

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        if entities.get("salary"):
            return {
                "salary": entities["salary"][0]
            }

        match = re.search(
            r"(\d[\d,]*)\s*(AED|USD|INR|₹|\$)?",
            query,
            re.IGNORECASE
        )

        if not match:
            return {}

        value = int(
            match.group(1).replace(",", "")
        )

        currency = match.group(2)

        return {
            "salary": {
                "value": value,
                "currency": currency
            }
        }