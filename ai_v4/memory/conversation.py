class ConversationMemory:
    def __init__(self, max_messages=20): self.max_messages=max_messages
    def append(self,memory,role,content):
        memory.setdefault("conversation",[]).append({"role":role,"content":content})
        memory["conversation"]=memory["conversation"][-self.max_messages:]
    def recent(self,memory,n=8): return memory.get("conversation",[])[-n:]
