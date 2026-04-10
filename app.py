"""
FastAPI HTTP server for CodeReview OpenEnv.

Exposes the environment as a REST API for agents to interact with.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.environment import CodeReviewEnv, TASK_SPECS
from env.models import Action, ReviewCategory, ReviewComment, Severity

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CodeReview OpenEnv",
    description="An OpenEnv-compliant AI training environment for Python code review.",
    version="1.0.0",
)

# In-memory session store
SESSIONS: Dict[str, CodeReviewEnv] = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_1_easy"
    session_id: str = "default"


class StepRequest(BaseModel):
    session_id: str = "default"
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def landing_page():
    """HTML landing page."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>CodeReview OpenEnv</title></head>
    <body style="font-family: system-ui, sans-serif; max-width: 720px; margin: 60px auto; padding: 0 20px;">
        <h1>🔍 CodeReview OpenEnv</h1>
        <p>An OpenEnv-compliant AI training environment for Python code review.</p>
        <h2>Endpoints</h2>
        <ul>
            <li><code>GET  /health</code> — Health check</li>
            <li><code>GET  /tasks</code> — List all task specs</li>
            <li><code>POST /reset</code> — Start or restart an episode</li>
            <li><code>POST /step</code> — Submit an action</li>
            <li><code>GET  /state</code> — Get full serialisable state</li>
            <li><code>GET  /docs</code> — Interactive Swagger UI</li>
        </ul>
    </body>
    </html>
    """


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Return specs for all available tasks."""
    return {
        task_id: spec.model_dump()
        for task_id, spec in TASK_SPECS.items()
    }


@app.post("/reset")
def reset(req: ResetRequest):
    """Start or restart an episode for the given task and session."""
    if req.task_id not in TASK_SPECS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Choose from: {list(TASK_SPECS.keys())}",
        )
    env = CodeReviewEnv(task_id=req.task_id)
    obs = env.reset()
    SESSIONS[req.session_id] = env
    return {"observation": obs.model_dump(), "session_id": req.session_id}


@app.post("/step")
def step(req: StepRequest):
    """Submit an action for the given session."""
    env = SESSIONS.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Call /reset first.",
        )

    # Parse the action dict into an Action model
    action_dict = req.action
    comments = []
    for c in action_dict.get("comments", []):
        try:
            comments.append(ReviewComment(
                line=c.get("line"),
                category=ReviewCategory(c.get("category", "bug")),
                severity=Severity(c.get("severity", "medium")),
                message=c.get("message", ""),
                suggestion=c.get("suggestion"),
            ))
        except Exception:
            pass  # skip malformed comments

    action = Action(
        comments=comments,
        summary=action_dict.get("summary"),
        submit=action_dict.get("submit", False),
    )

    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward.model_dump(),
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def get_state(session_id: str = Query(default="default")):
    """Return full serialisable state for the given session."""
    env = SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    return env.state().model_dump()
