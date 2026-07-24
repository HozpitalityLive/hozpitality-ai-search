from ai_v4.planner.intent import Intent


class PlannerRouter:

    ROUTES = {

        Intent.GREETING: {
            "agents": [],
            "need_search": False,
        },

        Intent.CHAT: {
            "agents": [],
            "need_search": False,
        },

        Intent.JOB_SEARCH: {
            "agents": ["job"],
            "need_search": True,
        },

        Intent.COMPANY_SEARCH: {
            "agents": ["company"],
            "need_search": True,
        },

        Intent.PROFESSIONAL_SEARCH: {
            "agents": ["professional"],
            "need_search": True,
        },

        Intent.ARTICLE_SEARCH: {
            "agents": ["article"],
            "need_search": True,
        },

        Intent.EVENT_SEARCH: {
            "agents": ["event"],
            "need_search": True,
        },

        Intent.AWARD_SEARCH: {
            "agents": ["award"],
            "need_search": True,
        },

        Intent.PRODUCT_SEARCH: {
            "agents": ["product"],
            "need_search": True,
        },

        Intent.FAQ: {
            "agents": ["faq"],
            "need_search": True,
        }

    }

    async def route(
        self,
        intent,
        entities
    ):

        route = self.ROUTES.get(intent)

        if not route:

            return {
                "mode": "single",
                "agents": [],
                "parallel": False,
                "need_search": False,
                "need_memory": True,
                "need_llm": True
            }

        return {
            "mode": "parallel" if len(route["agents"]) > 1 else "single",
            "parallel": len(route["agents"]) > 1,
            "need_memory": True,
            "need_llm": True,
            **route
        }