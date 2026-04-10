"""
Pydantic typed models for CodeReview OpenEnv.

Defines all core data structures: enums for review categories and severities,
code snippets, review comments, actions, observations, rewards, step results,
task specifications, and environment state.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReviewCategory(str, Enum):
    """Categories of code review issues."""
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    DOCUMENTATION = "documentation"


class Severity(str, Enum):
    """Severity levels for review comments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskDifficulty(str, Enum):
    """Difficulty levels for tasks."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class CodeSnippet(BaseModel):
    """A Python source code snippet for review."""
    file_name: str
    source: str
    language: str = "python"


class ReviewComment(BaseModel):
    """A single review comment identifying an issue in the code."""
    line: Optional[int] = None
    category: ReviewCategory
    severity: Severity = Severity.MEDIUM
    message: str
    suggestion: Optional[str] = None


class Action(BaseModel):
    """Agent action: a list of review comments plus control flags."""
    comments: List[ReviewComment] = Field(default_factory=list)
    summary: Optional[str] = None
    submit: bool = False


class Observation(BaseModel):
    """What the agent sees on each step."""
    task_id: str
    step: int
    snippet: CodeSnippet
    instructions: str
    previous_comments: List[ReviewComment] = Field(default_factory=list)
    feedback: Optional[str] = None
    done: bool = False


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""


class StepResult(BaseModel):
    """Result of a single environment step."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    """Specification for a single task."""
    task_id: str
    title: str
    difficulty: TaskDifficulty
    categories: List[str]
    description: str
    max_steps: int
    passing_threshold: float


class EnvironmentState(BaseModel):
    """Full serialisable state snapshot of the environment."""
    task_id: str
    step: int
    max_steps: int
    total_reward: float
    comments_so_far: List[ReviewComment] = Field(default_factory=list)
    done: bool
    grader_scores: Dict[str, Any] = Field(default_factory=dict)
