import asyncio

from ai_v4.agents.job_agent import JobAgent
from ai_v4.agents.company_agent import CompanyAgent
from ai_v4.agents.professional_agent import ProfessionalAgent

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
            # "faq": FaqAgent(),
        }

    async def execute(
        self,
        plan,
        query,
        memory
    ):

        logger.info("[2/4] Executing Agent(s)...")

        route = plan["route"]
        agent_names = route["agents"]

        tasks = []

        for agent_name in agent_names:

            agent = self.agents.get(agent_name)

            if not agent:
                raise Exception(
                    f"Unknown Agent {agent_name}"
                )

            tasks.append(
                agent.execute(
                    query=query,
                    plan=plan,
                    memory=memory
                )
            )

        results = await asyncio.gather(*tasks)

        if len(results) == 1:
            return results[0]

        merged = {
            "agent": ",".join(agent_names),
            "query": query,
            "filters": {},
            "page": 1,
            "page_size": 5,
            "total": 0,
            "results": []
        }

        for result in results:

            merged["filters"][result["agent"]] = result["filters"]

            merged["results"].extend(
                result["results"]
            )

            merged["total"] += result["total"]

        return merged