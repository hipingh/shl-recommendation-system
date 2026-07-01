"""
app/models/request_models.py

Pydantic request models for the /chat endpoint.
Schema is contractual — must not change without versioning.

Per SHL spec: the API is stateless. Every POST /chat call carries the
FULL conversation history. No session_id, no server-side persistence
required for the contract to be honored.
"""

from typing import List, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single turn in the conversation."""

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Who sent this message — 'user' or 'assistant'.",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The text content of the message.",
    )


class ChatRequest(BaseModel):
    """
    Payload for POST /chat.

    The caller sends the full conversation history on every call.
    The service is stateless and stores no per-conversation state.
    """

    messages: List[Message] = Field(
        ...,
        min_length=1,
        description="Full conversation history so far, oldest message first.",
    )
