from ai_v4.planner.intent import Intent
class PlannerRouter:
    ROUTES={
      Intent.JOB_SEARCH:["job"],Intent.JOB_ANALYTICS:["job"],Intent.COMPANY_SEARCH:["company"],
      Intent.PROFESSIONAL_SEARCH:["professional"],Intent.ARTICLE_SEARCH:["article"],
      Intent.EVENT_SEARCH:["event"],Intent.PRODUCT_SEARCH:["product"],Intent.AWARD_SEARCH:["awards"],Intent.FAQ:["faq"]
    }
    async def route(self,intent,entities):
        a=self.ROUTES.get(intent,[])
        return {"mode":"parallel" if len(a)>1 else "single","agents":a,"parallel":len(a)>1,"need_memory":True,"need_llm":True,"need_search":bool(a)}
