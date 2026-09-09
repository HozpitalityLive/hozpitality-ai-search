from enum import Enum
import re
from ai_v4.config.logger import logger

class Intent(str,Enum):
    GREETING="greeting"; CHAT="chat"; JOB_SEARCH="job_search"; COMPANY_SEARCH="company_search"
    PROFESSIONAL_SEARCH="professional_search"; ARTICLE_SEARCH="article_search"; EVENT_SEARCH="event_search"
    PRODUCT_SEARCH="product_search"; AWARD_SEARCH="award_search"; FAQ="faq"; UNKNOWN="unknown"
    JOB_ANALYTICS="job_analytics"; MULTI_SEARCH="multi_search"

class IntentDetector:
    def __init__(self):
        self.rules=[
            (Intent.GREETING,[r"^(hi|hello|hey|good morning|good evening)[!. ]*$"]),
            (Intent.JOB_ANALYTICS,[r"\bhow many\b.*\bjobs?\b",r"\bwhich companies\b.*\bhiring\b",r"\bjobs?\b.*\b(most|highest|average|count)\b"]),
            (Intent.ARTICLE_SEARCH,[r"\barticles?\b",r"\bnews\b",r"\bblogs?\b"]),
            (Intent.EVENT_SEARCH,[r"\bevents?\b",r"\bconference\b",r"\bexpo\b",r"\bsummit\b"]),
            (Intent.PRODUCT_SEARCH,[r"\bproducts?\b",r"\bmarketplace\b",r"\bbuy\b"]),
            (Intent.AWARD_SEARCH,[r"\bawards?\b",r"\bwinners?\b"]),
            (Intent.FAQ,[r"^how do i\b",r"^how can i\b",r"^what is\b",r"^why\b",r"\bsteps?\b"]),
            (Intent.PROFESSIONAL_SEARCH,[r"\bcandidates?\b",r"\bprofessionals?\b",r"\bdevelopers?\b",r"\bchefs?\b.*\b(best|candidates?)\b"]),
            (Intent.COMPANY_SEARCH,[r"\bcompanies?\b",r"\bhotels?\b",r"\brestaurants?\b",r"\btell me about\b"]),
            (Intent.JOB_SEARCH,[r"\bjobs?\b",r"\bvancanc(y|ies)\b",r"\bvacanc(y|ies)\b",r"\bhiring\b",r"\bcareers?\b",r"\bapply\b"]),
        ]
    async def detect(self,query):
        t=query.lower().strip()
        for intent,patterns in self.rules:
            if any(re.search(p,t) for p in patterns): return intent
        return Intent.CHAT
