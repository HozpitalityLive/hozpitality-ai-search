from ai_v4.llm.prompts.job_prompt import JobPrompt
from ai_v4.llm.prompts.chat_prompt import ChatPrompt
# from ai_v4.llm.prompts.company_prompt import CompanyPrompt
# from ai_v4.llm.prompts.professional_prompt import ProfessionalPrompt
# from ai_v4.llm.prompts.article_prompt import ArticlePrompt
# from ai_v4.llm.prompts.marketplace_prompt import MarketplacePrompt
# from ai_v4.llm.prompts.awards_prompt import AwardsPrompt
# from ai_v4.llm.prompts.event_prompt import EventPrompt
# from ai_v4.llm.prompts.faq_prompt import FAQPrompt
# from ai_v4.llm.prompts.admin_prompt import AdminPrompt


class PromptManager:

    def __init__(self):

        self.prompts = {
            "chat": ChatPrompt(),
            "job": JobPrompt(),
            # "company": CompanyPrompt(),
            # "professional": ProfessionalPrompt(),
            # "article": ArticlePrompt(),
            # "marketplace": MarketplacePrompt(),
            # "award": AwardsPrompt(),
            # "event": EventPrompt(),
            # "faq": FAQPrompt(),
            # "admin": AdminPrompt()
        }

    def build(
        self,
        agent,
        query,
        context,
        memory
    ):

        prompt = self.prompts.get(agent)

        if prompt is None:
            raise ValueError(f"No prompt registered for '{agent}'")

        return prompt.build(
            query=query,
            context=context,
            memory=memory
        )