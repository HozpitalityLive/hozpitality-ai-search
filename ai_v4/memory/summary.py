class MemorySummary:
    def summarize(self,memory):
        last=memory.get("last_search") or {}
        return {"last_query":last.get("query"),"last_agent":last.get("agent"),"last_filters":last.get("filters",{})}
