import calendar
import re
from datetime import date, datetime, timedelta

from .base import BaseParser


class DateParser(BaseParser):

    MONTHS = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        query_lower = query.lower()

        today = date.today()

        result = {
            "date": {
                "from": None,
                "to": None,
                "time_scope": None,
            }
        }

        
        if "today" in query_lower:
            result["date"]["from"] = today.isoformat()
            result["date"]["to"] = today.isoformat()
            result["time_scope"] = "today"

            return result

        if "tomorrow" in query_lower:
            d = today + timedelta(days=1)

            result["date"]["from"] = d.isoformat()
            result["date"]["to"] = d.isoformat()
            result["time_scope"] = "tomorrow"

            return result


        if "yesterday" in query_lower:
            d = today - timedelta(days=1)

            result["date"]["from"] = d.isoformat()
            result["date"]["to"] = d.isoformat()
            result["time_scope"] = "yesterday"

            return result

        
        if "this week" in query_lower:
            start = today - timedelta(days=today.weekday())
            end = start + timedelta(days=6)

            result["date"] = {
                "from": start.isoformat(),
                "to": end.isoformat(),
            }
            result["time_scope"] = "this_week"

            return result

        if "next week" in query_lower:
            start = today - timedelta(days=today.weekday()) + timedelta(days=7)
            end = start + timedelta(days=6)

            result["date"] = {
                "from": start.isoformat(),
                "to": end.isoformat(),
            }
            result["time_scope"] = "next_week"

            return result
        

        if "last week" in query_lower:
            start = today - timedelta(days=today.weekday()) - timedelta(days=7)
            end = start + timedelta(days=6)

            result["date"] = {
                "from": start.isoformat(),
                "to": end.isoformat(),
            }
            result["time_scope"] = "last_week"

            return result
        

        
        if "this month" in query_lower:
            start = today.replace(day=1)

            last_day = calendar.monthrange(
                today.year,
                today.month,
            )[1]

            end = today.replace(day=last_day)

            result["date"] = {
                "from": start.isoformat(),
                "to": end.isoformat(),
            }
            result["time_scope"] = "this_month"

            return result

        
        for month_name, month_number in self.MONTHS.items():
            if month_name in query_lower:
                year = today.year
                year_match = re.search(r"(20\d{2})", query)

                if year_match:
                    year = int(year_match.group(1))

                start = date(year, month_number, 1)

                last_day = calendar.monthrange(
                    year,
                    month_number,
                )[1]

                end = date(
                    year,
                    month_number,
                    last_day,
                )

                result["date"] = {
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                }
                result["time_scope"] = "month"

                return result

        
        year_match = re.search(
            r"\b(20\d{2})\b",
            query,
        )

        if year_match:
            year = int(year_match.group(1))
            result["date"] = {
                "from": f"{year}-01-01",
                "to": f"{year}-12-31",
            }
            result["time_scope"] = "year"

            return result

        
        if "latest" in query_lower or "recent" in query_lower:
            result["time_scope"] = "latest"

            return result


        if "upcoming" in query_lower:
            result["date"] = {
                "from": today.isoformat(),
            }

            result["time_scope"] = "upcoming"

            return result

        return {}