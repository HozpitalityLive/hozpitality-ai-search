from ai_v4.planner.extractors.regex_extractor import RegexExtractor
class LangExtractExtractor:
    def __init__(self): self.regex=RegexExtractor()
    async def extract(self,query): return self.regex.extract(query)
