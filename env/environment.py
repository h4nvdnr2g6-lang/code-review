"""
CodeReviewEnv – main OpenEnv environment.

Interface
---------
env = CodeReviewEnv(task_id="task_1_easy")
obs = env.reset()
result = env.step(action)
state = env.state()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from corpus.snippets import CORPUS
from env.models import (
    Action,
    CodeSnippet,
    EnvironmentState,
    Observation,
    Reward,
    ReviewComment,
    StepResult,
    TaskDifficulty,
    TaskSpec,
)
from graders.graders import GRADERS

# ---------------------------------------------------------------------------
# Task specs
# ---------------------------------------------------------------------------

TASK_SPECS: dict[str, TaskSpec] = {
    "task_1_easy": TaskSpec(
        task_id="task_1_easy",
        title="Bug Detection & Style Review",
        difficulty=TaskDifficulty.EASY,
        categories=["bug", "style"],
        description=(
            "Review calculator.py for correctness bugs (division by zero, off-by-one, "
            "empty collection crashes) and Python style issues. "
            "You do NOT need to check for security or performance."
        ),
        max_steps=5,
        passing_threshold=0.55,
    ),
    "task_2_medium": TaskSpec(
        task_id="task_2_medium",
        title="Security & Performance Audit",
        difficulty=TaskDifficulty.MEDIUM,
        categories=["security", "performance"],
        description=(
            "Audit user_service.py for security vulnerabilities (SQL injection, weak "
            "hashing, unsafe deserialization) and performance problems (unbounded queries, "
            "connection churn). Identify ALL critical security issues – missing one costs "
            "heavily."
        ),
        max_steps=7,
        passing_threshold=0.60,
    ),
    "task_3_hard": TaskSpec(
        task_id="task_3_hard",
        title="Comprehensive Code Review",
        difficulty=TaskDifficulty.HARD,
        categories=["bug", "security", "performance", "style", "documentation"],
        description=(
            "Perform a full production-grade review of data_pipeline.py covering bugs, "
            "security flaws, performance issues, code style, and documentation gaps. "
            "You MUST provide a written summary of overall findings. "
            "This snippet has intentional issues across all five categories."
        ),
        max_steps=10,
        passing_threshold=0.65,
    ),
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

INSTRUCTIONS_TEMPLATE = """
You are performing a Python code review.

Task: {title}
Difficulty: {difficulty}
Categories to check: {categories}

{description}

Your job:
1. Read the code snippet carefully.
2. Identify issues matching the specified categories.
3. For each issue, provide: line number (if applicable), category, severity, a clear message, and an optional fix suggestion.
4. When you are satisfied, set `submit=True` in your action.
{summary_note}

The code will be shown in the observation. Previous comments you have already submitted are also included so you can refine or expand them across steps.
""".strip()


class CodeReviewEnv:
    """
    OpenEnv-compliant environment for Python code review tasks.
    """

    def __init__(self, task_id: str = "task_1_easy"):
        if task_id not in TASK_SPECS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_SPECS)}")

        self.task_id = task_id
        self.spec: TaskSpec = TASK_SPECS[task_id]
        self.corpus_entry: dict = CORPUS[task_id]
        self.grader = GRADERS[task_id]
        self.ground_truth: List[ReviewComment] = self.corpus_entry["issues"]
        self.snippet: CodeSnippet = self.corpus_entry["snippet"]

        # State
        self._step: int = 0
        self._done: bool = False
        self._comments: List[ReviewComment] = []
        self._total_reward: float = 0.0
        self._grader_scores: Dict[str, float] = {}
        self._last_feedback: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment to initial state and return first observation."""
        self._step = 0
        self._done = False
        self._comments = []
        self._total_reward = 0.0
        self._grader_scores = {}
        self._last_feedback = None
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Advance the environment by one step.

        Parameters
        ----------
        action : Action
            Comments produced this step plus optional submit flag.

        Returns
        -------
        StepResult with (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")

        self._step += 1

        # Accumulate comments (deduplicate by message fingerprint)
        new_comments = self._deduplicate(action.comments)
        self._comments.extend(new_comments)

        # Compute incremental reward for new comments
        reward, feedback, grader_result = self._compute_reward(action, new_comments)
        self._grader_scores = grader_result
        self._total_reward = round(self._total_reward + reward.value, 4)
        self._last_feedback = feedback

        # Determine done
        done = action.submit or self._step >= self.spec.max_steps
        self._done = done

        obs = self._build_observation(feedback=feedback, done=done)
        info: Dict[str, Any] = {
            "step": self._step,
            "new_comments": len(new_comments),
            "total_comments": len(self._comments),
            "grader": grader_result,
            "passed": grader_result.get("score", 0.0) >= self.spec.passing_threshold,
        }

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> EnvironmentState:
        """Return full serialisable state snapshot."""
        return EnvironmentState(
            task_id=self.task_id,
            step=self._step,
            max_steps=self.spec.max_steps,
            total_reward=self._total_reward,
            comments_so_far=self._comments,
            done=self._done,
            grader_scores=self._grader_scores,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        feedback: Optional[str] = None,
        done: bool = False,
    ) -> Observation:
        summary_note = (
            "\n5. You MUST include a `summary` field with your overall assessment."
            if self.task_id == "task_3_hard"
            else ""
        )
        instructions = INSTRUCTIONS_TEMPLATE.format(
            title=self.spec.title,
            difficulty=self.spec.difficulty.value.upper(),
            categories=", ".join(self.spec.categories),
            description=self.spec.description,
            summary_note=summary_note,
        )
        return Observation(
            task_id=self.task_id,
            step=self._step,
            snippet=self.snippet,
            instructions=instructions,
            previous_comments=list(self._comments),
            feedback=feedback or self._last_feedback,
            done=done,
        )

    def _compute_reward(
        self,
        action: Action,
        new_comments: List[ReviewComment],
    ) -> tuple[Reward, str, dict]:
        """
        Compute reward with partial progress signals.

        Components
        ----------
        * +step_signal  : positive if new valid comments were added
        * +submit_bonus : grader score applied on final submit
        * -loop_penalty : penalty for submitting zero new comments repeatedly
        * -over_comment : penalty for > 2× the expected number of comments
        """
        # Run grader against ALL accumulated comments
        full_action = Action(
            comments=self._comments,
            summary=action.summary,
            submit=action.submit,
        )
        grader_result = self.grader.grade(full_action, self.ground_truth)
        current_score = grader_result["score"]

        breakdown: Dict[str, float] = {}
        reward_val = 0.0

        if action.submit:
            # Final reward = full grader score (0–1 mapped to -0.2–1.0)
            submit_reward = current_score * 0.8 + (0.2 if current_score >= self.spec.passing_threshold else -0.2)
            reward_val += submit_reward
            breakdown["submit_reward"] = round(submit_reward, 4)
            feedback = (
                f"Review submitted. Score: {current_score:.3f} "
                f"({'PASSED' if current_score >= self.spec.passing_threshold else 'FAILED'}). "
                f"Matched {grader_result['matched_count']}/{grader_result['total_ground_truth']} issues."
            )
        else:
            # Incremental reward: positive if new valid comments detected
            if new_comments:
                # Small positive signal for adding comments (+0.05 per comment, capped)
                step_reward = min(0.05 * len(new_comments), 0.15)
                reward_val += step_reward
                breakdown["step_reward"] = round(step_reward, 4)

                # Progress signal: reward increase in grader score
                # We run a "previous" grader check without new comments to get delta
                prev_action = Action(
                    comments=[c for c in self._comments if c not in new_comments],
                    summary=None,
                    submit=False,
                )
                prev_result = self.grader.grade(prev_action, self.ground_truth)
                score_delta = current_score - prev_result["score"]
                if score_delta > 0:
                    progress_reward = round(score_delta * 0.5, 4)
                    reward_val += progress_reward
                    breakdown["progress_reward"] = progress_reward
            else:
                # Penalty for empty step
                reward_val -= 0.05
                breakdown["empty_step_penalty"] = -0.05

            # Penalty for too many comments (spam)
            expected = grader_result["total_ground_truth"]
            if len(self._comments) > expected * 2.5:
                spam_penalty = -0.10
                reward_val += spam_penalty
                breakdown["spam_penalty"] = spam_penalty

            feedback = (
                f"Step {self._step}: Added {len(new_comments)} comment(s). "
                f"Running score: {current_score:.3f}. "
                f"Steps remaining: {self.spec.max_steps - self._step}."
            )

        reward_val = round(max(-1.0, min(1.0, reward_val)), 4)
        return Reward(value=reward_val, breakdown=breakdown, reason=feedback), feedback, grader_result

    def _deduplicate(self, incoming: List[ReviewComment]) -> List[ReviewComment]:
        """Remove comments whose (line, category, message[:40]) already exist."""
        existing_keys = {
            (c.line, c.category, c.message[:40]) for c in self._comments
        }
        new: List[ReviewComment] = []
        for c in incoming:
            key = (c.line, c.category, c.message[:40])
            if key not in existing_keys:
                existing_keys.add(key)
                new.append(c)
        return new
