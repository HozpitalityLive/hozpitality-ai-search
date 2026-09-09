import asyncio
from ai_v4.agents.job_agent import JobAgent
from ai_v4.agents.company_agent import CompanyAgent
from ai_v4.agents.professional_agent import ProfessionalAgent
from ai_v4.agents.article_agent import ArticleAgent
from ai_v4.agents.product_agent import ProductAgent
from ai_v4.agents.event_agent import EventAgent
from ai_v4.agents.awards_agent import AwardsAgent
from ai_v4.agents.faq_agent import FaqAgent
class AgentService:
    def __init__(self):
        self.agents={"job":JobAgent(),"company":CompanyAgent(),"professional":ProfessionalAgent(),"article":ArticleAgent(),"product":ProductAgent(),"event":EventAgent(),"awards":AwardsAgent(),"faq":FaqAgent()}
    async def execute(self,plan,query,memory):
        names=plan["route"]["agents"]
        if not names: return {"agent":plan["intent"],"query":query,"filters":{},"total":0,"page":1,"page_size":5,"results":[]}
        out=await asyncio.gather(*(self.agents[n].execute(query,plan,memory) for n in names))
        if len(out)==1:return out[0]
        merged={"agent":",".join(names),"query":query,"filters":{},"total":0,"page":1,"page_size":5,"results":[]}
        for x in out: merged["filters"][x["agent"]]=x["filters"];merged["results"]+=x["results"];merged["total"]+=x["total"]
        return merged
