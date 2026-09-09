from ai_v4.context.formatter import ContextFormatter
class ContextBuilder:
    def __init__(self): self.formatter=ContextFormatter()
    async def build(self,query,search_results,memory=None):
        docs=[x.get("document",{}) for x in search_results]
        cats=list(dict.fromkeys([d.get("category") or d.get("category_text") for d in docs if d.get("category") or d.get("category_text")]))[:5]
        locs=list(dict.fromkeys([d.get("location") or d.get("location_text") for d in docs if d.get("location") or d.get("location_text")]))[:5]
        comps=list(dict.fromkeys([d.get("user_name") or d.get("company") for d in docs if d.get("user_name") or d.get("company")]))[:5]
        return {"query":query,"memory":memory,"summary":{"query":query,"total":len(docs),"categories":cats,"top_locations":locs,"top_companies":comps},"documents":self.formatter.format_documents(search_results[:8]),"results":search_results,"total":len(docs)}
