class AwardsFilterBuilder:
    def build(self,plan):
        e=plan.get("entities",{}); f={"category":"awards"}
        if e.get("locations"): f["locations"]=e["locations"]
        if e.get("skills"): f["skills"]=e["skills"]
        if e.get("companies"): f["companies"]=e["companies"]
        if e.get("job_titles"): f["roles"]=e["job_titles"]
        if e.get("exclude_terms"): f["exclude_terms"]=e["exclude_terms"]
        if e.get("date"): f["date"]=e["date"]
        return f
