"""
Framework-agnostic chat handling logic.
"""

import uuid
import re
from typing import AsyncGenerator, List, Optional

from ...components import UiComponent, RichTextComponent, SimpleTextComponent
from ...core import Agent
from .models import ChatRequest, ChatResponse, ChatStreamChunk


class ChatHandler:
    """Core chat handling logic - framework agnostic."""

    def __init__(
        self,
        agent: Agent,
    ):
        """Initialize chat handler.

        Args:
            agent: The agent to handle chat requests
        """
        self.agent = agent

    async def handle_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        """Stream chat responses.

        Args:
            request: Chat request

        Yields:
            Chat stream chunks
        """
        conversation_id = request.conversation_id or self._generate_conversation_id()
        # Use request_id from client for tracking, or use the one generated internally
        request_id = request.request_id or str(uuid.uuid4())

        # Fast-path common conversational messages. These do not invoke the
        # LLM, database, Chroma memory, or tool loop.
        greeting = self._get_fast_response(request.message)
        if greeting:
            # Convert the component to the actual ChatStreamChunk expected by
            # the FastAPI SSE/JSON routes. Yielding UiComponent directly here
            # causes the frontend to receive no `rich`/`simple` payload.
            component = UiComponent(
                rich_component=RichTextComponent(content=greeting, markdown=True),
                simple_component=SimpleTextComponent(text=greeting),
            )
            yield ChatStreamChunk.from_component(
                component, conversation_id, request_id
            )
            return

        clarification = self._get_clarification_response(request.message)
        if clarification:
            component = UiComponent(
                rich_component=RichTextComponent(content=clarification, markdown=True),
                simple_component=SimpleTextComponent(text=clarification),
            )
            yield ChatStreamChunk.from_component(
                component, conversation_id, request_id
            )
            return

        async for component in self.agent.send_message(
            request_context=request.request_context,
            message=request.message,
            conversation_id=conversation_id,
        ):
            yield ChatStreamChunk.from_component(component, conversation_id, request_id)

    async def handle_poll(self, request: ChatRequest) -> ChatResponse:
        """Handle polling-based chat.

        Args:
            request: Chat request

        Returns:
            Complete chat response
        """
        chunks = []
        async for chunk in self.handle_stream(request):
            chunks.append(chunk)

        return ChatResponse.from_chunks(chunks)

    @staticmethod
    def _get_fast_response(message: str) -> Optional[str]:
        """Return an immediate response for common conversational messages."""
        text = " ".join((message or "").strip().lower().split())
        text = re.sub(r"[^\\w\\s]", "", text)
        if not text:
            return None

        responses = {
            "hi": "Hi! 👋 How can I help you with Hozpitality?",
            "hello": "Hello! 👋 How can I help you today?",
            "hey": "Hey! 👋 What can I help you find?",
            "hi there": "Hi there! 👋 What would you like to find on Hozpitality?",
            "hello there": "Hello! 👋 What can I help you with?",
            "hey there": "Hey there! 👋 How can I help?",
            "good morning": "Good morning! ☀️ How can I help you today?",
            "good afternoon": "Good afternoon! 👋 How can I help you today?",
            "good evening": "Good evening! 🌙 How can I help you today?",
            "how are you": "I'm doing well, thanks! I'm ready to help you search Hozpitality.",
            "thanks": "You're welcome! 😊",
            "thank you": "You're welcome! 😊",
            "who are you": "I'm Hozpitality AI. I can help you search and analyze Hozpitality data.",
            "what can you do": "I can help you find jobs and answer questions using Hozpitality's live data.",
            "help": "Try asking: “Find waiter jobs” or “How many available jobs are there?”",
            "hi vanna": "Hi! 👋 How can I help you with Hozpitality?",
            "hello vanna": "Hello! 👋 How can I help you today?",
            "good day": "Hello! 👋 How can I help you today?",
            "nice to meet you": "Nice to meet you too! 👋 What would you like to find?",
            "what is hozpitality ai": "I'm Hozpitality AI. I can help you search jobs and analyze live Hozpitality data.",
            "are you there": "Yes, I'm here! 👋 What would you like to find?",
        }

        return responses.get(text)

    @staticmethod
    def _get_clarification_response(message: str) -> Optional[str]:
        """Fast QA gate for under-specified search requests.

        This is deterministic and local so clarification never incurs an LLM
        round-trip. Clear requests pass directly to Vanna.
        """
        text = " ".join((message or "").strip().lower().split())
        if not text:
            return "What would you like me to search for?"

        generic_jobs = {
            "job", "jobs", "find job", "find jobs", "search job",
            "search jobs", "show jobs", "show me jobs", "find a job",
            "find me a job", "looking for a job",
        }
        if text in generic_jobs:
            return "What type of job would you like me to find? You can also include a location, for example: **waiter jobs in Dubai**."

        generic_data = {"search", "find", "show me data", "show data", "find data", "search something", "find something"}
        if text in generic_data:
            return "What would you like me to search for? Please give me the type of record or topic you need."

        job_words = ("job", "jobs", "vacancy", "vacancies", "opening", "openings", "career", "careers")
        search_words = ("find", "search", "show", "looking", "look for", "get", "give me")
        role_indicators = (
            "waiter", "waitress", "chef", "cook", "housekeeping", "reception",
            "front office", "manager", "supervisor", "bartender", "steward",
            "engineer", "accountant", "sales", "hr", "human resources",
            "security", "driver", "concierge", "bellman", "intern", "developer",
            "marketing", "finance", "purchasing", "maintenance", "technician",
        )
        if any(w in text for w in job_words) and any(w in text for w in search_words) and not any(r in text for r in role_indicators):
            return "What type of job should I search for? You can give me a role and, if needed, a location—for example: **chef jobs in Dubai**."

        return None

    def _generate_conversation_id(self) -> str:
        """Generate new conversation ID."""
        return f"conv_{uuid.uuid4().hex[:8]}"
