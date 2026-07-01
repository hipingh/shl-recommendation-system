"""
app/models/response_models.py

Pydantic response models for the /chat endpoint.
Schema is contractual — must not change without versioning.
"""

from typing import List

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    """
    A single SHL assessment recommendation.

    Every URL must be sourced from catalog.json — never hallucinated.
    """

    name: str = Field(..., description="Official SHL assessment name.")
    url: str = Field(..., description="Direct SHL product catalog URL.")
    test_type: str = Field(
        ...,
        description=(
            "Category of the assessment, e.g. "
            "'Knowledge & Skills', 'Personality & Behavior', "
            "'Simulations', 'Competencies'."
        ),
    )


class ChatResponse(BaseModel):
    """
    Response payload for POST /chat.

    This schema is fixed — consumers depend on it exactly as specified:
    reply, recommendations, end_of_conversation. No session_id — the
    service is stateless.
    """

    reply: str = Field(
        ...,
        description="The agent's natural-language response to the user.",
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description=(
            "Ordered list of recommended SHL assessments (1–10). "
            "Empty when intent is CLARIFY, COMPARE reply, or REFUSAL."
        ),
    )
    end_of_conversation: bool = Field(
        default=False,
        description=(
            "True when the agent has nothing further to clarify or recommend "
            "and the conversation can be considered complete."
        ),
    )
    
