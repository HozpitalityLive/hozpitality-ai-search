from fastapi import WebSocket
import json,re
from ai_v4.llm.brain import Brain
class ResponseGenerator:
    def __init__(self): self.brain=Brain()
    async def generate(self,websocket,agent,query,context,memory=None,model=None):
        await websocket.send_json({"type":"thinking","message":"Thinking..."})
        raw=""
        async for chunk in self.brain.think(agent,query,context,memory,model):
            raw+=chunk.get("response","")
        ai=self._parse(raw,agent)
        await websocket.send_json({"type":"intro","intent":ai["intent"],"content":ai["intro"]})
        await websocket.send_json({"type":"description","content":ai["description"]})
        await websocket.send_json({"type":"results","intent":ai["intent"],"total":context.get("total",0),"results":context.get("results",[])})
        await websocket.send_json({"type":"follow_up","questions":ai["follow_up"]})
        await websocket.send_json({"type":"done"})
        return ai
    def _parse(self,raw,agent):
        s=raw.strip();s=re.sub(r"^```json|^```|```$","",s).strip()
        try:
            a=json.loads(s[s.find("{"):s.rfind("}")+1])
            return {"intent":a.get("intent",agent),"intro":a.get("intro") or a.get("response",""),"description":a.get("description",""),"follow_up":(a.get("follow_up") or [])[:3]}
        except Exception:
            return {"intent":agent,"intro":"I found matching results.","description":"You can review the results below and refine the search if needed.","follow_up":["Show more results","Filter by location","Filter by seniority"]}
