from ai_v4.agents.job_agent import JobAgent
from ai_v4.agents.company_agent import CompanyAgent
from ai_v4.agents.professional_agent import ProfessionalAgent
from ai_v4.agents.article_agent import ArticleAgent
from ai_v4.agents.product_agent import ProductAgent
from ai_v4.agents.event_agent import EventAgent
from ai_v4.agents.awards_agent import AwardsAgent
from ai_v4.agents.faq_agent import FaqAgent
from ai_v4.config.logger import logger


class AgentService:

    def __init__(self):

        self.agents = {
            "job": JobAgent(),
            "company": CompanyAgent(),
            "professional": ProfessionalAgent(),
            # "article": ArticleAgent(),
            # "product": ProductAgent(),
            # "event": EventAgent(),
            # "award": AwardsAgent(),
            # "faq": FaqAgent()
        }

    async def execute(
        self,
        plan,
        query,
        memory
    ):
        
        logger.info("[2/4] Executing agent...")

        route = plan["route"]
        agent_name = route["agents"][0]
        agent = self.agents.get(agent_name)

        if not agent:
            raise Exception(
                f"Unknown Agent {agent_name}"
            )

        return await agent.execute(
            query=query,
            plan=plan,
            memory=memory
        )