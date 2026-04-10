"""
Test suite for CodeReview OpenEnv.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import CodeReviewEnv
from env.models import Action, ReviewCategory, ReviewComment, Severity
from graders.graders import Task1Grader, Task2Grader, Task3Grader
from corpus.snippets import CORPUS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def perfect_action(task_id: str) -> Action:
    """Build an action containing all ground-truth comments for a task."""
    issues = CORPUS[task_id]["issues"]
    return Action(comments=list(issues), summary="Perfect review.", submit=True)


def empty_action(submit: bool = False) -> Action:
    return Action(comments=[], submit=submit)


def single_bug_action() -> Action:
    return Action(
        comments=[
            ReviewComment(
                line=2,
                category=ReviewCategory.BUG,
                severity=Severity.HIGH,
                message="divide() has no guard against division by zero will raise ZeroDivisionError",
                suggestion="Add a check for b==0",
            )
        ],
        submit=True,
    )


# ---------------------------------------------------------------------------
# Grader unit tests
# ---------------------------------------------------------------------------

class TestTask1Grader:
    grader = Task1Grader()
    ground_truth = CORPUS["task_1_easy"]["issues"]

    def test_perfect_score_close_to_one(self):
        action = perfect_action("task_1_easy")
        result = self.grader.grade(action, self.ground_truth)
        assert result["score"] >= 0.80, f"Expected ≥0.80 got {result['score']}"

    def test_empty_action_scores_zero(self):
        result = self.grader.grade(empty_action(submit=True), self.ground_truth)
        assert result["score"] < 0.15

    def test_single_correct_bug_gives_positive_score(self):
        result = self.grader.grade(single_bug_action(), self.ground_truth)
        assert result["score"] > 0.0

    def test_wrong_category_penalised(self):
        action = Action(
            comments=[
                ReviewComment(
                    line=2, category=ReviewCategory.SECURITY,
                    severity=Severity.HIGH,
                    message="divide has no guard against division by zero",
                )
            ],
            submit=True,
        )
        result_wrong = self.grader.grade(action, self.ground_truth)
        result_right = self.grader.grade(single_bug_action(), self.ground_truth)
        assert result_right["score"] >= result_wrong["score"]

    def test_fabricated_comment_penalised(self):
        fabricated = Action(
            comments=[
                ReviewComment(
                    line=5, category=ReviewCategory.BUG,
                    severity=Severity.CRITICAL,
                    message="Imaginary crash that does not exist in the code at all",
                )
            ] * 10,
            submit=True,
        )
        result = self.grader.grade(fabricated, self.ground_truth)
        assert result["score"] <= 0.1

    def test_score_in_range(self):
        action = perfect_action("task_1_easy")
        result = self.grader.grade(action, self.ground_truth)
        assert 0.0 <= result["score"] <= 1.0


class TestTask2Grader:
    grader = Task2Grader()
    ground_truth = CORPUS["task_2_medium"]["issues"]

    def test_perfect_score_close_to_one(self):
        action = perfect_action("task_2_medium")
        result = self.grader.grade(action, self.ground_truth)
        assert result["score"] >= 0.75

    def test_missing_critical_sql_injection_penalised(self):
        # Remove the SQL injection comment from perfect action
        issues = [i for i in self.ground_truth
                  if not ("SQL injection" in i.message or "injection" in i.message.lower())]
        action = Action(comments=issues, submit=True)
        full_action = perfect_action("task_2_medium")
        full_result = self.grader.grade(full_action, self.ground_truth)
        partial_result = self.grader.grade(action, self.ground_truth)
        assert full_result["score"] > partial_result["score"]

    def test_score_in_range(self):
        action = perfect_action("task_2_medium")
        result = self.grader.grade(action, self.ground_truth)
        assert 0.0 <= result["score"] <= 1.0


class TestTask3Grader:
    grader = Task3Grader()
    ground_truth = CORPUS["task_3_hard"]["issues"]

    def test_perfect_with_summary_beats_without(self):
        with_summary = perfect_action("task_3_hard")
        without_summary = Action(
            comments=list(self.ground_truth), summary=None, submit=True
        )
        r_with = self.grader.grade(with_summary, self.ground_truth)
        r_without = self.grader.grade(without_summary, self.ground_truth)
        assert r_with["score"] >= r_without["score"]

    def test_summary_penalty_applied_when_missing(self):
        action = Action(comments=[], summary=None, submit=True)
        result = self.grader.grade(action, self.ground_truth)
        assert result["breakdown"].get("summary_penalty", 0) < 0

    def test_score_in_range(self):
        action = perfect_action("task_3_hard")
        result = self.grader.grade(action, self.ground_truth)
        assert 0.0 <= result["score"] <= 1.0


# ---------------------------------------------------------------------------
# Environment integration tests
# ---------------------------------------------------------------------------

class TestEnvironmentAPI:
    def test_reset_returns_observation(self):
        env = CodeReviewEnv("task_1_easy")
        obs = env.reset()
        assert obs.task_id == "task_1_easy"
        assert obs.step == 0
        assert obs.snippet.language == "python"
        assert len(obs.snippet.source) > 0

    def test_step_increments_step_counter(self):
        env = CodeReviewEnv("task_1_easy")
        env.reset()
        result = env.step(empty_action(submit=False))
        assert result.observation.step == 1

    def test_step_submit_ends_episode(self):
        env = CodeReviewEnv("task_1_easy")
        env.reset()
        result = env.step(empty_action(submit=True))
        assert result.done is True

    def test_step_after_done_raises(self):
        env = CodeReviewEnv("task_1_easy")
        env.reset()
        env.step(empty_action(submit=True))
        with pytest.raises(RuntimeError):
            env.step(empty_action())

    def test_state_matches_step(self):
        env = CodeReviewEnv("task_2_medium")
        env.reset()
        env.step(single_bug_action())
        state = env.state()
        assert state.step == 1
        assert state.task_id == "task_2_medium"

    def test_max_steps_auto_terminates(self):
        env = CodeReviewEnv("task_1_easy")
        env.reset()
        result = None
        for _ in range(env.spec.max_steps):
            result = env.step(empty_action(submit=False))
        assert result.done is True

    def test_reward_in_range(self):
        env = CodeReviewEnv("task_1_easy")
        env.reset()
        result = env.step(single_bug_action())
        assert -1.0 <= result.reward.value <= 1.0

    def test_reset_clears_state(self):
        env = CodeReviewEnv("task_1_easy")
        env.reset()
        env.step(single_bug_action())
        env.reset()
        state = env.state()
        assert state.step == 0
        assert state.total_reward == 0.0
        assert len(state.comments_so_far) == 0

    def test_deduplication_prevents_duplicate_comments(self):
        env = CodeReviewEnv("task_1_easy")
        env.reset()
        # First step: submit=False so episode stays open
        step1_action = Action(comments=[
            ReviewComment(
                line=2, category=ReviewCategory.BUG, severity=Severity.HIGH,
                message="divide() has no guard against division by zero will raise ZeroDivisionError",
                suggestion="Add a check for b==0",
            )
        ], submit=False)
        env.step(step1_action)
        # Second step: same comment again (should be deduped)
        step2_action = Action(comments=[
            ReviewComment(
                line=2, category=ReviewCategory.BUG, severity=Severity.HIGH,
                message="divide() has no guard against division by zero will raise ZeroDivisionError",
                suggestion="Add a check for b==0",
            )
        ], submit=True)
        env.step(step2_action)
        state = env.state()
        assert len(state.comments_so_far) == 1

    def test_all_three_tasks_init(self):
        for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            env = CodeReviewEnv(tid)
            obs = env.reset()
            assert obs.task_id == tid

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            CodeReviewEnv("task_9_impossible")

    def test_hard_task_requires_summary_field(self):
        env = CodeReviewEnv("task_3_hard")
        env.reset()
        # Submit without summary – should still work but score less
        action = Action(comments=[], summary=None, submit=True)
        result = env.step(action)
        assert result.done is True
        # Verify summary penalty is applied
        assert result.info["grader"]["breakdown"].get("summary_penalty", 0) < 0

    def test_full_episode_task1(self):
        """Full happy-path episode: submit all ground truth → should pass."""
        env = CodeReviewEnv("task_1_easy")
        env.reset()
        action = perfect_action("task_1_easy")
        result = env.step(action)
        assert result.done
        assert result.info["passed"] is True
