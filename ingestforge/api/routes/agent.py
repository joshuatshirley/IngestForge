"""
Agent API Router.

Intelligence Briefs
Exposes agentic research synthesis capabilities and manages research missions.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #9: Complete type hints.
"""

import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from ingestforge.agent.brief_models import IntelligenceBrief
from ingestforge.agent.brief_generator import IFBriefGenerator, BriefCitationValidator
from ingestforge.core.config_loaders import load_config
from ingestforge.llm.factory import get_llm_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/agent", tags=["agent"])

# =============================================================================
# SCHEMAS
# =============================================================================


class AgentRequest(BaseModel):
    task: str
    max_steps: int = 10
    provider: str = "llamacpp"
    roadmap: Optional[List[dict]] = None


class AgentMissionStatus(BaseModel):
    id: str
    task: str
    status: str
    steps: List[dict]
    tasks: Optional[List[dict]] = None
    answer: Optional[str] = None
    verification: Optional[dict] = None
    progress: int = 0


class BriefRequest(BaseModel):
    """Request for intelligence brief generation."""

    mission_id: str
    query: str
    validate_citations: bool = True


class BriefResponse(BaseModel):
    """Response for intelligence brief generation."""

    brief: IntelligenceBrief
    markdown: str
    validation_passed: bool = True


# =============================================================================
# STATE (Shared with main.py - needs a proper service/store eventually)
# =============================================================================
# For now, we import the shared job store from main to maintain state consistency
# In a real refactor, this would move to a Persistence Service.
from ingestforge.api.main import agent_jobs, get_current_user, _enforce_job_limit

# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/run")
async def run_agent_task(
    req: AgentRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    _enforce_job_limit()
    job_id = str(uuid.uuid4())
    agent_jobs[job_id] = {
        "id": job_id,
        "task": req.task,
        "status": "RUNNING",
        "steps": [],
        "tasks": req.roadmap,
        "answer": None,
        "progress": 0,
    }
    # execute_agent_mission is still in main.py for now to avoid circular deps
    from ingestforge.api.main import execute_agent_mission

    background_tasks.add_task(execute_agent_mission, job_id, req)
    return {"job_id": job_id}


@router.post("/plan")
async def generate_plan(req: AgentRequest):
    from ingestforge.agent.strategist.generator import RoadmapGenerator

    config = load_config()
    llm_client = get_llm_client(config, req.provider)
    generator = RoadmapGenerator(llm_client)
    roadmap = generator.generate(req.task)
    return roadmap.dict()


@router.get("/status/{job_id}", response_model=AgentMissionStatus)
def get_agent_status(job_id: str, current_user: dict = Depends(get_current_user)):
    if job_id not in agent_jobs:
        raise HTTPException(status_code=404, detail="Mission not found")
    return agent_jobs[job_id]


@router.post("/generate-brief", response_model=BriefResponse)
async def generate_brief(
    request: BriefRequest, current_user: dict = Depends(get_current_user)
) -> BriefResponse:
    """
    Generate a boardroom-ready intelligence brief.

    Backend API entry point for G-RAG synthesis.
    """
    try:
        generator = IFBriefGenerator()
        brief = await generator.generate_brief(request.mission_id, request.query)

        validation_ok = True
        if request.validate_citations:
            validator = BriefCitationValidator(generator.pipeline.storage)
            validation_ok = validator.validate(brief)

        return BriefResponse(
            brief=brief, markdown=brief.to_markdown(), validation_passed=validation_ok
        )

    except Exception as e:
        logger.exception("Brief generation failed")
        raise HTTPException(status_code=500, detail=str(e))
