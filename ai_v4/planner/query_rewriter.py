import re
class QueryRewriter:
    async def rewrite(self,original_query,clarification_answer,intent=None,entities=None):
        return re.sub(r"\s+"," ",f"{original_query} {clarification_answer}").strip()
