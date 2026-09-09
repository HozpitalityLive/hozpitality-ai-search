class JobFilterBuilder:
    def build(self,plan):
        e=plan.get("entities",{}); f={"category":"job"}
        for src,dst in [("locations","locations"),("skills","skills"),("job_titles","roles"),("exclude_terms","exclude_terms"),("employment_types","employment_types"),("experience","experience"),("salary","salary"),("date","date"),("job_level","job_level")]:
            if e.get(src): f[dst]=e[src]
        return f
