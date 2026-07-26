from ai_v4.config.logger import logger
from ai_v4.planner.intent import Intent

class SearchPlanner:

    async def build(
        self,
        intent: Intent,
        filters: dict
    ) -> dict:

        logger.info("[3/4] Building Search Plan...")

        plan = {
            "can_search": True,
            "filters": filters,
            "missing": [],
            "reason": None
        }

        if intent == Intent.PROFESSIONAL_SEARCH:

            if not (
                filters.get("name")
                or filters.get("keyword")
                or filters.get("skill")
                or filters.get("designation")
                or filters.get("location")
            ):
                plan["can_search"] = False
                plan["missing"].append("professional")

        elif intent == Intent.JOB_SEARCH:

            if not (
                filters.get("job_title")
                or filters.get("keyword")
                or filters.get("location")
            ):
                plan["can_search"] = False
                plan["missing"].append("job_title")

        elif intent == Intent.COMPANY_SEARCH:

            if not (
                filters.get("company")
                or filters.get("keyword")
                or filters.get("location")
            ):
                plan["can_search"] = False
                plan["missing"].append("company")

        elif intent == Intent.EVENT_SEARCH:

            if not (
                filters.get("event")
                or filters.get("keyword")
                or filters.get("location")
            ):
                plan["can_search"] = False
                plan["missing"].append("event")

        elif intent == Intent.AWARD_SEARCH:

            if not (
                filters.get("award")
                or filters.get("keyword")
            ):
                plan["can_search"] = False
                plan["missing"].append("award")

        elif intent == Intent.MARKETPLACE_SEARCH:

            if not filters.get("keyword"):
                plan["can_search"] = False
                plan["missing"].append("product")

        elif intent == Intent.ARTICLE_SEARCH:

            if not filters.get("keyword"):
                plan["can_search"] = False
                plan["missing"].append("article")

        logger.info("=" * 80)
        logger.info("SEARCH PLAN")
        logger.info(plan)
        logger.info("=" * 80)

        return plan