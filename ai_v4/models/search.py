class SearchResult(BaseModel):
    engine: str
    score: float
    document: dict