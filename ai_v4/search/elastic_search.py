import asyncio
from ai_v4.config.elastic import es
from ai_v4.config.settings import settings
class ElasticSearch:
    def __init__(self): self.index=settings.ELASTIC_INDEX
    def _search(self,query,filters,size):
        must=[{"multi_match":{"query":query,"fields":["title^5","ai_keywords^6","content^2","location^3","user_name^2"],"operator":"or","fuzziness":"AUTO"}}]
        filters_es=[]
        for loc in filters.get("locations",[]):
            vals=["Dubai","Abu Dhabi","Sharjah","United Arab Emirates","UAE"] if loc.lower()=="uae" else [loc]
            filters_es.append({"bool":{"should":[{"match":{"location":v}} for v in vals],"minimum_should_match":1}})
        for ex in filters.get("exclude_terms",[]):
            filters_es.append({"bool":{"must_not":{"multi_match":{"query":ex,"fields":["title","content","ai_keywords"]}}}})
        cat=filters.get("category")
        if cat: filters_es.append({"term":{"category":cat}})
        body={"size":size,"query":{"bool":{"must":must,"filter":filters_es}}}
        try:r=es.search(index=self.index,body=body)
        except Exception:return []
        return [{"engine":"elastic","score":h.get("_score",0),"document":{"id":h.get("_id"),**h.get("_source",{})}} for h in r["hits"]["hits"]]
    async def search(self,query,filters=None,size=20):
        return await asyncio.to_thread(self._search,query,filters or {},size) if settings.USE_ELASTIC else []
