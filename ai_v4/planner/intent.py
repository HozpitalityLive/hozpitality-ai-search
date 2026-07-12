from enum import Enum
from ai_v4.config.logger import logger


class Intent(str, Enum):

    GREETING = "greeting"
    CHAT = "chat"
    JOB_SEARCH = "job_search"
    COMPANY_SEARCH = "company_search"
    PROFESSIONAL_SEARCH = "professional_search"
    ARTICLE_SEARCH = "article_search"
    EVENT_SEARCH = "event_search"
    PRODUCT_SEARCH = "product_search"
    AWARD_SEARCH = "award_search"
    FAQ = "faq"
    UNKNOWN = "unknown"


class IntentDetector:

    def __init__(self):
        self.intent_rules = {
            Intent.GREETING: [
                "hi",
                "hello",
                "hey",
                "good morning",
                "good evening"
            ],

            Intent.JOB_SEARCH: [
                "job",
                "jobs",
                "vacancy",
                "vacancies",
                "career",
                "careers",
                "hiring",
                "recruitment",
                "apply"
            ],

            Intent.COMPANY_SEARCH: [
                "company",
                "companies",
                "hotel",
                "hotels",
                "restaurant",
                "restaurants"
            ],

            Intent.PROFESSIONAL_SEARCH: [
                "candidate",
                "professional",
                "employee",
                "staff",
                "chef",
                "manager"
            ],

            Intent.ARTICLE_SEARCH: [
                "article",
                "news",
                "blog"
            ],

            Intent.EVENT_SEARCH: [
                "event",
                "conference",
                "expo",
                "summit"
            ],

            Intent.PRODUCT_SEARCH: [
                "product",
                "marketplace",
                "buy",
                "supplier"
            ],

            Intent.AWARD_SEARCH: [
                "award",
                "awards",
                "winner"
            ],

            Intent.FAQ: [
                "how",
                "what",
                "why",
                "when",
                "guide",
                "steps"
            ]
        }

    async def detect(
        self,
        query: str
    ) -> Intent:

        text = query.lower().strip()

        logger.info(
            f"Detecting Intent : {text}"
        )

        for intent, keywords in self.intent_rules.items():
            for keyword in keywords:
                if keyword in text:
                    return intent

        return Intent.CHAT