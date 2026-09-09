from ai_v4.llm.prompts.job_prompt import JobPrompt
from ai_v4.llm.prompts.chat_prompt import ChatPrompt
from ai_v4.llm.prompts.company_prompt import CompanyPrompt
from ai_v4.llm.prompts.professional_prompt import ProfessionalPrompt
from ai_v4.llm.prompts.article_prompt import ArticlePrompt
from ai_v4.llm.prompts.marketplace_prompt import MarketplacePrompt
from ai_v4.llm.prompts.awards_prompt import AwardsPrompt
from ai_v4.llm.prompts.event_prompt import EventPrompt
from ai_v4.llm.prompts.faq_prompt import FAQPrompt
class PromptManager:
    def __init__(self):
        self.prompts={"job":JobPrompt(),"company":CompanyPrompt(),"professional":ProfessionalPrompt(),"article":ArticlePrompt(),"marketplace":MarketplacePrompt(),"product":MarketplacePrompt(),"award":AwardsPrompt(),"awards":AwardsPrompt(),"event":EventPrompt(),"faq":FAQPrompt(),"chat":ChatPrompt()}
    def build(self,agent,query,context,memory=None):
        p=self.prompts.get(agent,self.prompts["chat"])
        return p.build(query,context,memory)
