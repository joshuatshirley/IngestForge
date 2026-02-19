"""
IngestForge API Bridge.

Bridges the React Web Portal to the IngestForge Engine.
Follows NASA JPL Rules #1, #4, and #9.

Task 125: Security hardening (HSTS, CSP).
Task 126: Prometheus monitoring integration.
"""

from fastapi import (
    FastAPI,
    Depends,
    BackgroundTasks,
    UploadFile,
    File,
    HTTPException,
    status,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import uuid
import logging
import os

# JPL Rule #2: Fixed upper bounds
MAX_AGENT_JOBS = 1000

# Module logger
logger = logging.getLogger(__name__)

from ingestforge.core.config_loaders import load_config
from ingestforge.core.auth.service import AuthService
from ingestforge.agent.react_engine import create_engine
from ingestforge.agent.llm_adapter import create_llm_think_adapter
from ingestforge.llm.factory import get_llm_client
from ingestforge.llm.base import GenerationConfig
from ingestforge.api.middleware.nexus_auth import NexusAuthMiddleware

from ingestforge.api.routes.nexus_mgmt import router as nexus_mgmt_router

app = FastAPI(title="IngestForge API", version="1.2.0")

# ... existing code ...

app.include_router(nexus_mgmt_router)

# --- Middleware ---
app.add_middleware(NexusAuthMiddleware)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Enforce Strict-Transport-Security and Content-Security-Policy (Task 125).
    """
    response: Response = await call_next(request)
    response.headers[
        "Strict-Transport-Security"
    ] = "max-age=31536000; includeSubDomains"
    response.headers[
        "Content-Security-Policy"
    ] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self';"
    return response


# --- State ---
agent_jobs: Dict[str, dict] = {}


def _enforce_job_limit() -> None:
    """Enforce MAX_AGENT_JOBS limit. Rule #2: Fixed bounds."""
    if len(agent_jobs) >= MAX_AGENT_JOBS:
        # Remove oldest 10% of jobs
        to_remove = len(agent_jobs) - int(MAX_AGENT_JOBS * 0.9)
        oldest_keys = list(agent_jobs.keys())[:to_remove]
        for key in oldest_keys:
            del agent_jobs[key]
        logger.info(f"Cleaned up {to_remove} old jobs to enforce limit")


# --- Auth Dependency ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="v1/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = AuthService.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


# --- Shared Task Workers ---


async def execute_agent_mission(job_id: str, req: Any) -> None:
    """Shared worker for agent missions.

    Rule #9: Complete type hints.
    """
    from ingestforge.api.routes.ws_status import notify_job_update

    try:
        from ingestforge.agent.strategist.executor import RoadmapExecutor
        from ingestforge.agent.strategist.models import ResearchRoadmap

        config = load_config()
        llm_client = get_llm_client(config, req.provider)
        think_adapter = create_llm_think_adapter(
            llm_client=llm_client,
            config=GenerationConfig(stop_sequences=["Observation:"]),
        )
        engine = create_engine(think_fn=think_adapter, max_iterations=req.max_steps)

        if req.roadmap:
            executor = RoadmapExecutor(engine)
            roadmap = ResearchRoadmap(objective=req.task, tasks=req.roadmap)

            for i, task in enumerate(roadmap.tasks):
                agent_jobs[job_id]["tasks"][i]["status"] = "in_progress"
                await notify_job_update(job_id, agent_jobs[job_id])
                result = engine.run(task.description)
                agent_jobs[job_id]["tasks"][i]["status"] = (
                    "completed" if result.success else "failed"
                )
                agent_jobs[job_id]["tasks"][i]["result_summary"] = result.final_answer
                agent_jobs[job_id]["steps"].extend([s.to_dict() for s in result.steps])
                await notify_job_update(job_id, agent_jobs[job_id])

            agent_jobs[job_id]["status"] = "COMPLETED"
            agent_jobs[job_id][
                "answer"
            ] = "Mission complete. Structured roadmap results synthesized above."
        else:
            result = engine.run(req.task)
            agent_jobs[job_id]["status"] = "COMPLETED" if result.success else "FAILED"
            agent_jobs[job_id]["answer"] = result.final_answer
            agent_jobs[job_id]["steps"] = [s.to_dict() for s in result.steps]

        await notify_job_update(job_id, agent_jobs[job_id])

    except Exception as e:
        agent_jobs[job_id]["status"] = "FAILED"
        agent_jobs[job_id]["error"] = str(e)
        await notify_job_update(job_id, agent_jobs[job_id])


# --- Routers ---

# Include synthesis router ()
from ingestforge.api.routes.synthesis import router as synthesis_router

app.include_router(synthesis_router)

# Include health router (US-BUG-003)
from ingestforge.api.routes.health import router as health_router

app.include_router(health_router)

# Include learning router ()
from ingestforge.api.routes.learning import router as learning_router

app.include_router(learning_router)

# Include WebSocket status router ()
from ingestforge.api.routes.ws_status import router as ws_status_router

app.include_router(ws_status_router)

# Include Streaming Foundry router ()
from ingestforge.api.routes.foundry import router as foundry_router

app.include_router(foundry_router)

# Include Query Clarification router ()
from ingestforge.api.routes.query import router as query_router

app.include_router(query_router)

# Include Evidence Links router ()
from ingestforge.api.routes.evidence import router as evidence_router

app.include_router(evidence_router)

# Include Discovery router ()
from ingestforge.api.routes.discovery import router as discovery_router

app.include_router(discovery_router)

# Include Connectors router ()
from ingestforge.api.routes.connectors import router as connectors_router

app.include_router(connectors_router)

# Include Chat router ()
from ingestforge.api.routes.chat import router as chat_router

app.include_router(chat_router)

# Include Legal Vertical router ()
from ingestforge.api.routes.legal import router as legal_router

app.include_router(legal_router)

# Include Timeline router ()
from ingestforge.api.routes.timeline import router as timeline_router

app.include_router(timeline_router)

# Include Agent router (Intelligence Briefs & Missions)
from ingestforge.api.routes.agent import router as agent_router

app.include_router(agent_router)

# Include Nexus router (Task 127 - Federation Handshake)
from ingestforge.api.routes.nexus import router as nexus_router

app.include_router(nexus_router)

# Include Maintenance router ()
from ingestforge.api.routes.maintenance import router as maintenance_router

app.include_router(maintenance_router)


@app.get("/metrics")
def get_metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas for non-routed endpoints ---


class Token(BaseModel):
    access_token: str
    token_type: str


class RemoteIngestRequest(BaseModel):
    platform: str
    source_id: str
    token: Optional[str] = None


class TransformRequest(BaseModel):
    operation: str
    target_library: str = "default"
    params: Dict[str, Any] = {}


# --- Root Ingestion Routes (Still in main for now) ---


@app.post("/v1/ingest/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    _enforce_job_limit()
    job_id = str(uuid.uuid4())
    agent_jobs[job_id] = {
        "id": job_id,
        "task": f"File: {file.filename}",
        "status": "PENDING",
        "progress": 0,
    }
    return {"job_id": job_id}


@app.post("/v1/ingest/remote")
async def ingest_remote_document(
    req: RemoteIngestRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    _enforce_job_limit()
    job_id = str(uuid.uuid4())
    agent_jobs[job_id] = {
        "id": job_id,
        "task": f"Cloud: {req.platform} ({req.source_id})",
        "status": "PENDING",
        "progress": 0,
    }
    background_tasks.add_task(run_remote_ingest_worker, job_id, req)
    return {"job_id": job_id}


async def run_remote_ingest_worker(job_id: str, req: RemoteIngestRequest) -> None:
    """Process remote ingest job. Rule #9: Complete type hints."""
    from ingestforge.api.routes.ws_status import notify_job_update

    logger.info(f"Remote Ingest Job {job_id} started.")
    agent_jobs[job_id]["status"] = "RUNNING"
    await notify_job_update(job_id, agent_jobs[job_id])


@app.get("/v1/ingest/status/{job_id}")
def get_job_status(job_id: str, current_user: dict = Depends(get_current_user)):
    return agent_jobs.get(job_id, {"status": "UNKNOWN"})


# --- Sync & Transform Logic ---


@app.post("/v1/sync/push")
async def trigger_sync_push(
    background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)
):
    job_id = str(uuid.uuid4())
    return {"job_id": job_id, "status": "SYNC_STARTED"}


@app.post("/v1/corpus/transform")
async def transform_corpus(
    req: TransformRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    _enforce_job_limit()
    job_id = str(uuid.uuid4())
    agent_jobs[job_id] = {
        "id": job_id,
        "task": f"Transform: {req.operation}",
        "status": "RUNNING",
        "progress": 0,
    }
    return {"job_id": job_id}


@app.post("/v1/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user with environment-configured credentials.
    """
    admin_username = os.getenv("INGESTFORGE_ADMIN_USERNAME")
    admin_password_hash = os.getenv("INGESTFORGE_ADMIN_PASSWORD_HASH")

    if not admin_username or not admin_password_hash:
        logger.error("Authentication credentials not configured in environment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured. Contact administrator.",
        )

    if not form_data.username or len(form_data.username.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    if not form_data.password or len(form_data.password) == 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    if form_data.username != admin_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    if not AuthService.verify_password(form_data.password, admin_password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    access_token = AuthService.create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/v1/system/telemetry")
def get_telemetry(current_user: dict = Depends(get_current_user)):
    """Retrieve real-time hardware resource telemetry (Task 13.1.3)."""
    from ingestforge.core.system.hardware import HardwareDetector

    detector = HardwareDetector()
    snapshot = detector.get_snapshot()
    recommendation = detector.recommend_preset()

    return {
        "ram": {
            "total": snapshot.total_ram_gb,
            "available": snapshot.available_ram_gb,
            "usage_percent": round(
                (
                    (snapshot.total_ram_gb - snapshot.available_ram_gb)
                    / snapshot.total_ram_gb
                )
                * 100,
                1,
            ),
        },
        "cpu": {
            "count": snapshot.cpu_count,
            "usage_percent": snapshot.cpu_usage_percent,
        },
        "disk": {"free": snapshot.disk_free_gb},
        "recommendation": recommendation,
    }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the API server."""
    if not host or len(host.strip()) == 0:
        raise ValueError("Invalid host: must be non-empty")
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port} (must be 1-65535)")
    uvicorn.run("ingestforge.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
