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
            yield UiComponent(
                rich_component=RichTextComponent(content=greeting, markdown=True),
                simple_component=SimpleTextComponent(text=greeting),
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
        }

        return responses.get(text)

    def _generate_conversation_id(self) -> str:
        """Generate new conversation ID."""
        return f"conv_{uuid.uuid4().hex[:8]}"
