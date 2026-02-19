"""
Chat API Router.

Conversational Query Mode
Exposes /v1/chat endpoint for multi-turn RAG.

JPL Compliance:
- Rule #4: All functions < 60 lines
- Rule #7: Input validation via Pydantic
- Rule #9: Complete type hints
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ingestforge.chat.service import ChatService, ChatMessage as ServiceMessage
from ingestforge.api.routes.retrieval import (
    SearchResultItem,
    _map_search_result_to_item,
)

router = APIRouter(prefix="/v1", tags=["chat"])

# =============================================================================
# CONSTANTS (JPL Rule #2)
# =============================================================================

MAX_INPUT_MESSAGES = 50

# =============================================================================
# MODELS
# =============================================================================


class Message(BaseModel):
    """Chat message model."""

    role: str = Field(..., pattern="^(user|ai|system)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Request for chat turn.

    JPL Rule #2: Bounded input list.
    """

    messages: List[Message] = Field(
        ..., max_length=MAX_INPUT_MESSAGES, description="Conversation history"
    )
    broadcast: bool = Field(
        default=False, description="Whether to query remote Nexus peers"
    )
    nexus_ids: Optional[List[str]] = Field(
        default=None, description="Specific peer IDs to target"
    )

    @property
    def history(self) -> List[ServiceMessage]:
        """Convert to service messages (all but last)."""
        return [
            ServiceMessage(role=m.role, content=m.content) for m in self.messages[:-1]
        ]

    @property
    def last_query(self) -> str:
        """Get the latest user query."""
        if not self.messages or self.messages[-1].role != "user":
            raise ValueError("Last message must be from user")
        return self.messages[-1].content


class ChatResponse(BaseModel):
    """Response for chat turn."""

    answer: str
    sources: List[SearchResultItem]
    context_query: str


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/chat", response_model=ChatResponse, summary="Execute conversational RAG turn"
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat message with retrieval and history.

    Backend entry point.
    """
    try:
        service = ChatService()

        # Execute chat logic
        result = service.chat(history=request.history, user_query=request.last_query)

        # Map domain results to API model
        api_sources = [_map_search_result_to_item(r) for r in result.sources]

        return ChatResponse(
            answer=result.answer,
            sources=api_sources,
            context_query=result.context_query,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}",
        )
