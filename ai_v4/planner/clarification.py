class ClarificationDetector:
    async def analyze(self,query,intent,search_plan):
        q=query.lower()
        # These requests refer to an object that is not present in the message.
        if "this job" in q and ("similar" in q or "candidate" in q or "candidates" in q):
            return {"required":True,"question":"Please provide the job link or job reference number.","missing":["job"],"reason":"The target job is not identified."}
        if "this candidate" in q or "this professional" in q:
            return {"required":True,"question":"Please provide the candidate or professional profile link.","missing":["professional"],"reason":"The target professional is not identified."}
        if search_plan.get("can_search"): return {"required":False,"question":None,"missing":[],"reason":None}
        if intent.value=="job_search": q="What job role or position are you looking for?"
        elif intent.value=="company_search": q="Which company, hotel, or location are you interested in?"
        elif intent.value=="professional_search": q="Which professional role, skill, or location should I search?"
        else: q="What exactly would you like me to search for?"
        return {"required":True,"question":q,"missing":search_plan.get("missing",[]),"reason":search_plan.get("reason")}
