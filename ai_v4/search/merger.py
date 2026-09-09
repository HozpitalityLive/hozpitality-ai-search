class SearchMerger:
    def merge(self,*groups):
        out={}; scores={}
        for group in groups:
            for item in group:
                d=item.get("document",{}); key=str(d.get("object_id") or d.get("id") or (d.get("title"),d.get("slug")))
                if key not in out: out[key]=item
                scores[key]=max(scores.get(key,-1),float(item.get("score") or 0))
        return sorted(out.values(),key=lambda x:scores.get(str(x["document"].get("object_id") or x["document"].get("id") or (x["document"].get("title"),x["document"].get("slug"))),0),reverse=True)
