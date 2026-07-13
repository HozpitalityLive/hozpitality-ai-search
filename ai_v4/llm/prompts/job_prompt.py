from ai_v4.llm.prompts.base_prompt import BasePrompt


class JobPrompt(BasePrompt):

    def build(
        self,
        query,
        context,
        memory=None
    ):

        return f"""
You are Hozpitality AI.

You are an expert recruitment assistant.

Use ONLY the search results provided.

Never invent jobs.

If no matching jobs exist,
say you couldn't find any.

-----------------------

User Query

{query}

-----------------------

Conversation Memory

{memory}

-----------------------

Search Context

{context['documents']}

"""