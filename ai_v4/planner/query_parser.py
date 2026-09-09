class QueryParser:
    async def parse(self,query,intent,entities):
        e=entities or {}
        return {
            "keyword": " ".join(dict.fromkeys(e.get("skills",[])+e.get("job_titles",[])+e.get("articles",[])+e.get("events",[]))) or query,
            "locations": e.get("locations",[]),
            "roles": e.get("job_titles",[]),
            "skills": e.get("skills",[]),
            "salary": e.get("salary",[]),
            "experience": e.get("experience",[]),
            "date": e.get("date",{}),
            "job_level": e.get("job_level",[]),
            "employment_types": e.get("employment_types",[]),
            "exclude_terms": e.get("exclude_terms",[]),
            "company": e.get("companies",[]),
            "category": {
                "job_search":"job","job_analytics":"job","company_search":"company",
                "professional_search":"professional","article_search":"article",
                "event_search":"event","product_search":"product","award_search":"awards","faq":"faq"
            }.get(intent.value)
        }
