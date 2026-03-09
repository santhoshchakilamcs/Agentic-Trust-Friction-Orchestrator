"""
Real-Time API for the Agentic Risk & Retention Orchestrator.

Exposes the full agentic pipeline as a REST API so transactions
can be evaluated in real time with Claude reasoning.

Security layers:
  1. API Key authentication (X-API-Key header)
  2. Rate limiting (slowapi — protects Anthropic API credits)
  3. CORS policy (configurable allowed origins)
  4. PII hashing (SHA-256 before data reaches LLM agents)

Usage:
    uvicorn agentic_orchestrator.api:app --reload --port 8000

Then POST to http://localhost:8000/api/v1/evaluate
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from agentic_orchestrator.memory.short_term import TransactionState
from agentic_orchestrator.memory.long_term import LongTermMemory
from agentic_orchestrator.orchestrator.engine import OrchestratorEngine
from agentic_orchestrator.data.generator import UserProfile, generate_transactions
from agentic_orchestrator.llm.client import is_available
from agentic_orchestrator.compliance.pii_masker import PIIMasker

load_dotenv()

# ── Security Configuration ──────────────────────────────────────────────────
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate the X-API-Key header. Skips check if no key is configured."""
    if not API_SECRET_KEY:
        return  # No key configured — development mode
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# Rate limiter — 30 evaluate calls/minute, 120 general calls/minute
limiter = Limiter(key_func=get_remote_address)

# ── Pydantic Models ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """Incoming transaction to evaluate."""
    txn_id: str = Field(default="", description="Transaction ID (auto-generated if empty)")
    user_id: str = Field(..., description="User/sender ID")
    amount_usd: float = Field(..., gt=0, description="Transfer amount in USD")
    recipient_name: str = Field(..., description="Recipient name")
    recipient_country: str = Field(..., description="2-letter country code")
    corridor: str = Field(..., description="Sending-Receiving corridor, e.g. US-PH")
    device_id: str = Field(default="UNKNOWN", description="Device identifier")
    ip_address: str = Field(default="0.0.0.0", description="Sender IP address")
    timestamp: str = Field(default="", description="ISO timestamp (auto-filled if empty)")
    is_new_recipient: bool = Field(default=False, description="First time sending to this person?")

class EvaluationResponse(BaseModel):
    """Result of the agentic evaluation."""
    txn_id: str
    final_action: str
    risk_score: float
    risk_reasoning: str
    investigator_score: float
    investigator_flags: List[str]
    context_adjustment: float
    context_findings: List[str]
    challenge_message: str
    hitl_required: bool
    judge_score: Optional[float]
    judge_grade: str
    judge_feedback: str
    compliance_override: Optional[str]
    compliance_flags: List[str]
    processing_log: List[str]
    llm_enabled: bool

class UserProfileRequest(BaseModel):
    """Register a user profile for context-aware analysis."""
    user_id: str
    name: str
    country: str
    typical_amount: float
    typical_recipients: List[str] = []
    typical_corridor: str = ""
    registered_device_id: str = ""
    registered_ip_prefix: str = ""
    life_events: List[dict] = []

class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    model: str
    registered_users: int

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic Risk & Retention Orchestrator",
    description="Real-time fraud detection with Claude-powered multi-agent reasoning",
    version="2.0.0",
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Global state — initialized on startup
memory = LongTermMemory()
engine = OrchestratorEngine(memory, hitl_enabled=False, judge_enabled=True)
registered_user_ids: set = set()


@app.on_event("startup")
def startup():
    """Seed with sample users on startup."""
    _, users = generate_transactions(n_total=20, fraud_ratio=0.15, n_users=5)
    memory.ingest_user_profiles(users)
    for u in users:
        registered_user_ids.add(u.user_id)
    auth_mode = "API_KEY" if API_SECRET_KEY else "OPEN (no key configured)"
    print(f"✅ Orchestrator ready | LLM: {is_available()} | Users: {len(users)} | Auth: {auth_mode}")
    print(f"   Registered user IDs: {sorted(registered_user_ids)}")


# ── Static Files & UI ───────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the web UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
@limiter.limit("120/minute")
def health(request: Request):
    from agentic_orchestrator.llm.client import get_model
    return HealthResponse(
        status="ok",
        llm_available=is_available(),
        model=get_model(),
        registered_users=len(registered_user_ids),
    )


@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
@limiter.limit("30/minute")
def evaluate_transaction(
    request: Request,
    req: TransactionRequest,
    _key: None = Depends(verify_api_key),
):
    """Evaluate a single transaction through the full agentic pipeline.

    PII fields (ip_address, device_id, recipient_name) are SHA-256 hashed
    before being sent to the LLM agents. The original values are never
    stored or logged.
    """
    import uuid

    txn_id = req.txn_id or f"TXN-{uuid.uuid4().hex[:10].upper()}"
    timestamp = req.timestamp or datetime.now().isoformat()

    # ── PII hashing: hash sensitive fields before they reach the engine ──
    hashed_recipient = PIIMasker._hash(req.recipient_name, prefix="recipient")
    hashed_ip = PIIMasker._hash(req.ip_address, prefix="ip")
    hashed_device = PIIMasker._hash(req.device_id, prefix="device")

    state = TransactionState(
        txn_id=txn_id,
        user_id=req.user_id,
        amount_usd=req.amount_usd,
        recipient_name=hashed_recipient,
        recipient_country=req.recipient_country,  # needed for sanctions check
        corridor=req.corridor,
        device_id=hashed_device,
        ip_address=hashed_ip,
        timestamp=timestamp,
        is_new_recipient=req.is_new_recipient,
        pii_masked=True,
    )

    result = engine.process_transaction(state)

    return EvaluationResponse(
        txn_id=result.txn_id,
        final_action=result.final_action,
        risk_score=result.risk_score,
        risk_reasoning=result.risk_reasoning,
        investigator_score=result.investigator_score,
        investigator_flags=result.investigator_flags,
        context_adjustment=result.context_score_adjustment,
        context_findings=result.context_findings,
        challenge_message=result.challenge_message,
        hitl_required=result.hitl_required,
        judge_score=result.judge_score,
        judge_grade=result.judge_reasoning_grade,
        judge_feedback=result.judge_feedback,
        compliance_override=result.compliance_override,
        compliance_flags=result.compliance_flags,
        processing_log=result.processing_log,
        llm_enabled=is_available(),
    )


@app.post("/api/v1/register-user")
@limiter.limit("30/minute")
def register_user(
    request: Request,
    req: UserProfileRequest,
    _key: None = Depends(verify_api_key),
):
    """Register a new user profile for context-aware analysis."""
    profile = UserProfile(
        user_id=req.user_id,
        name=req.name,
        country=req.country,
        typical_amount=req.typical_amount,
        typical_recipients=req.typical_recipients,
        typical_corridor=req.typical_corridor,
        registered_device_id=req.registered_device_id,
        registered_ip_prefix=req.registered_ip_prefix,
        life_events=req.life_events,
    )
    memory.ingest_user_profiles([profile])
    registered_user_ids.add(req.user_id)
    return {"status": "registered", "user_id": req.user_id}


@app.get("/api/v1/users")
@limiter.limit("60/minute")
def list_users(
    request: Request,
    _key: None = Depends(verify_api_key),
):
    """List all registered user IDs."""
    return {"users": sorted(registered_user_ids), "count": len(registered_user_ids)}

