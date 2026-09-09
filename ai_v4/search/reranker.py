import re
class SearchReranker:
    def rerank(self,query,results):
        terms=set(re.findall(r"\w+",query.lower()))
        scored=[]
        for r in results:
            d=r["document"]; text=" ".join(str(d.get(k,"")) for k in ("title","content","ai_keywords","location_text","user_name")).lower()
            lexical=sum(1 for t in terms if len(t)>2 and t in text)
            score=float(r.get("score") or 0)+lexical*0.15
            r=dict(r);r["score"]=score;scored.append(r)
        return sorted(scored,key=lambda x:x["score"],reverse=True)
