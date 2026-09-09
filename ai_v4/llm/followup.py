class FollowUpGenerator:
    def generate(self,intent,filters):
        if intent=="job_search": return ["Show only senior jobs","Filter by location","Show higher-paying jobs"]
        if intent=="professional_search": return ["Filter by experience","Filter by location","Show matching skills"]
        if intent=="company_search": return ["Show companies hiring","Filter by location","Show hotel companies"]
        return ["Show more results","Filter by location","Refine the search"]
