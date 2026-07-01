"""
app/services/agent.py

The central AI agent orchestrator.

This is the brain of the system — it coordinates all services in the
correct order:

1. Reconstruct conversation state from the messages sent in the request.
2. Classify intent.
3. Route to the appropriate engine:
   - CLARIFY / UNKNOWN → ClarificationService
   - RECOMMEND / REFINE → RetrievalService + RecommendationService
   - COMPARE            → ComparisonService
   - OFF_TOPIC / PROMPT_INJECTION → RefusalService

The service is fully stateless: every call carries the entire
conversation history in request.messages, and nothing is persisted
between calls. SessionService is no longer used in the request path.
"""

import logging

from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse
from app.services.clarification_service import ClarificationService
from app.services.comparison_service import ComparisonService
from app.services.conversation_service import ConversationService
from app.services.intent_service import IntentService
from app.services.recommendation_service import RecommendationService
from app.services.refusal_service import RefusalService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Central AI agent that routes requests through the RAG pipeline.

    Wires together all sub-services via constructor injection.
    Each service has a single, well-defined responsibility.
    """

    def __init__(
        self,
        intent_service: IntentService,
        conversation_service: ConversationService,
        retrieval_service: RetrievalService,
        recommendation_service: RecommendationService,
        comparison_service: ComparisonService,
        clarification_service: ClarificationService,
        refusal_service: RefusalService,
    ) -> None:
        self._intent = intent_service
        self._conversation = conversation_service
        self._retrieval = retrieval_service
        self._recommendation = recommendation_service
        self._comparison = comparison_service
        self._clarification = clarification_service
        self._refusal = refusal_service

    def process(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request through the full agent pipeline.

        Args:
            request: ChatRequest containing the full conversation history
                      (request.messages), oldest message first.

        Returns:
            ChatResponse with reply, recommendations, and end_of_conversation flag.
        """
        # ── Step 1: Use the messages sent by the caller directly ──
        # No session lookup, no DB write — the caller is the source of truth.
        messages = request.messages

        # ── Step 2: Classify intent ───────────────────────────
        classification = self._intent.classify(messages)
        intent = classification.intent
        logger.info("Processing intent: %s | turns=%d", intent, len(messages))

        # ── Step 3: Route by intent and build response ────────
        if intent in ("OFF_TOPIC", "PROMPT_INJECTION"):
            response = self._refusal.refuse(intent)
        elif intent == "COMPARE":
            state = self._conversation.reconstruct_state(messages)
            response = self._handle_compare(classification.assessment_names, state)
        elif intent in ("RECOMMEND", "REFINE"):
            state = self._conversation.reconstruct_state(messages)
            conversation_history = self._conversation.format_for_prompt(messages)
            response = self._handle_recommend(state, conversation_history)
        else:  # CLARIFY or UNKNOWN
            state = self._conversation.reconstruct_state(messages)
            conversation_history = self._conversation.format_for_prompt(messages)
            response = self._handle_clarify(state, conversation_history)

        return response

    # ── Intent Handlers ───────────────────────────────────────

    def _handle_clarify(
        self,
        state,
        conversation_history: str,
    ) -> ChatResponse:
        """Ask a targeted clarification question."""
        # Even if intent is CLARIFY, if we have enough context, recommend instead
        if state.has_enough_context():
            logger.info(
                "State has enough context despite CLARIFY intent — recommending."
            )
            return self._handle_recommend(state, conversation_history)

        return self._clarification.clarify(state, conversation_history)

    def _handle_recommend(
        self,
        state,
        conversation_history: str,
    ) -> ChatResponse:
        """Retrieve assessments and generate ranked recommendations."""
        if not state.has_enough_context():
            # Not enough info — fall back to clarification
            logger.info("Insufficient context for RECOMMEND — falling back to CLARIFY.")
            return self._clarification.clarify(state, conversation_history)

        retrieved_docs = self._retrieval.retrieve(state)
        return self._recommendation.recommend(state, retrieved_docs, conversation_history)

    def _handle_compare(
        self,
        assessment_names: list,
        state,
    ) -> ChatResponse:
        """Fetch both named assessments and generate a comparison."""
        if len(assessment_names) < 2:
            # Not enough names identified — ask which two to compare
            return ChatResponse(
                reply=(
                    "I'd be happy to compare two assessments! "
                    "Could you specify which two SHL assessments you'd like to compare? "
                    'For example: "Compare OPQ32 vs Verify Numerical Reasoning."'
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        return self._comparison.compare(
            assessment_name_a=assessment_names[0],
            assessment_name_b=assessment_names[1],
            state=state,
        )


