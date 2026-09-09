import re
from datetime import date, timedelta
class RegexExtractor:
    def extract(self,q):
        t=q.lower(); e={"query":q,"person_names":[],"companies":[],"locations":[],"skills":[],"technologies":[],"job_titles":[],"departments":[],"experience":[],"salary":[],"awards":[],"events":[],"articles":[],"languages":[],"nationalities":[],"employment_types":[],"visas":[],"exclude_terms":[],"raw_entities":[]}
        locs=["dubai","abu dhabi","sharjah","ajman","ras al khaimah","uae","united arab emirates","mumbai","delhi","india","london","singapore","qatar","saudi arabia"]
        for x in locs:
            if re.search(r"\b"+re.escape(x)+r"\b",t): e["locations"].append(x.title() if x!="uae" else "UAE")
        roles=["chef","executive chef","sous chef","hotel manager","general manager","python developer","developer","waiter","housekeeping"]
        for x in roles:
            if re.search(r"\b"+re.escape(x)+r"\b",t): e["job_titles"].append(x)
        skills=["python","javascript","react","ai","machine learning","recruitment","hospitality","hotel"]
        for x in skills:
            if re.search(r"\b"+re.escape(x)+r"\b",t): e["skills"].append(x)
        m=re.search(r"(?:aed|usd|inr|₹|\$)\s*([\d,]+)",t)
        if not m: m=re.search(r"([\d,]+)\s*(?:aed|usd|inr|₹|\$)",t)
        if m: e["salary"].append({"value":int(m.group(1).replace(",","")),"currency":(re.search(r"(aed|usd|inr|₹|\$)",m.group(0)) or [None,None])[1]})
        m=re.search(r"(\d+)\+?\s*(?:years?|yrs?)",t)
        if m: e["experience"].append({"min":int(m.group(1)),"unit":"years"})
        for x in ["full-time","part-time","contract","internship"]:
            if x in t: e["employment_types"].append(x)
        if "not housekeeping" in t or "but not housekeeping" in t: e["exclude_terms"].append("housekeeping")
        e["date"]={}
        if "today" in t: e["date"]={"from":date.today().isoformat(),"to":date.today().isoformat(),"time_scope":"today"}
        if "yesterday" in t:
            d=date.today()-timedelta(days=1);e["date"]={"from":d.isoformat(),"to":d.isoformat(),"time_scope":"yesterday"}
        if "this week" in t:
            s=date.today()-timedelta(days=date.today().weekday());e["date"]={"from":s.isoformat(),"to":(s+timedelta(days=6)).isoformat(),"time_scope":"this_week"}
        if "senior" in t or "lead" in t or "head" in t: e["job_level"]=["senior"]
        return e
