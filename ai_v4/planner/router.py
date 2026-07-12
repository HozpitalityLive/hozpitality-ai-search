from ai_v4.planner.intent import Intent

class PlannerRouter:

    async def route(
        self,
        intent: Intent,
        entities: dict
    ) -> dict:

        route = {
            "mode": "single",
            "agents": ["general"],
            "parallel": False,
            "need_search": False,
            "need_memory": True,
            "need_llm": True
        }

        if intent == Intent.GREETING:
            return route

        if intent == Intent.JOB_SEARCH:
            route["agents"] = ["job"]
            route["need_search"] = True
            return route

        if intent == Intent.COMPANY_SEARCH:
            route["agents"] = ["company"]
            route["need_search"] = True
            return route

        if intent == Intent.PROFESSIONAL_SEARCH:
            route["agents"] = ["professional"]
            route["need_search"] = True
            return route

        if intent == Intent.PRODUCT_SEARCH:
            route["agents"] = ["marketplace"]
            route["need_search"] = True
            return route

        if intent == Intent.EVENT_SEARCH:
            route["agents"] = ["event"]
            route["need_search"] = True
            return route

        if intent == Intent.AWARD_SEARCH:
            route["agents"] = ["award"]
            route["need_search"] = True
            return route

        if intent == Intent.ARTICLE_SEARCH:
            route["agents"] = ["article"]
            route["need_search"] = True
            return route

        if intent == Intent.FAQ:
            route["agents"] = ["faq"]
            route["need_search"] = True
            return route

        return route