from ai_v4.planner.intent import Intent
class SearchPlanner:
    async def build(self,intent,filters):
        if intent in (Intent.GREETING,Intent.CHAT): return {"can_search":False,"missing":[],"reason":"chat"}
        if intent==Intent.FAQ: return {"can_search":True,"missing":[],"reason":None}
        if intent==Intent.JOB_ANALYTICS: return {"can_search":True,"missing":[],"reason":None}
        useful=any(filters.get(k) for k in ("keyword","locations","roles","skills","company","date"))
        return {"can_search":useful,"missing":[] if useful else ["search subject"],"reason":None if useful else "insufficient search subject"}
