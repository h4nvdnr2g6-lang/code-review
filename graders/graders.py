"""
Agent graders for all three tasks.

Each grader implements:
    grade(action: Action, ground_truth: list[ReviewComment]) -> dict

Scoring philosophy
------------------
* True positive (found real issue)         → positive reward
* False positive (fabricated issue)        → small penalty
* Missed critical issue                    → large penalty
* Summary quality (task 3)                → bonus
* Partial credit for correct category/severity with wrong line
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from env.models import Action, ReviewCategory, ReviewComment, Severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.CRITICAL: 1.0,
    Severity.HIGH: 0.75,
    Severity.MEDIUM: 0.5,
    Severity.LOW: 0.25,
}


def _category_match(a: ReviewComment, b: ReviewComment) -> bool:
    return a.category == b.category


def _severity_close(a: ReviewComment, b: ReviewComment) -> bool:
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    return abs(order.index(a.severity) - order.index(b.severity)) <= 1


def _line_close(a: ReviewComment, b: ReviewComment, tolerance: int = 3) -> bool:
    if a.line is None or b.line is None:
        return True  # file-level comments always match positionally
    return abs(a.line - b.line) <= tolerance


def _message_relevant(comment: ReviewComment, truth: ReviewComment) -> bool:
    """Check if comment message contains keywords from the truth message."""
    # Pull significant words (>4 chars) from the ground truth message
    truth_keywords = {
        w.lower()
        for w in re.findall(r"\b\w{4,}\b", truth.message)
        if w.lower() not in {"this", "that", "with", "from", "will", "should", "must", "have", "been", "when"}
    }
    comment_text = (comment.message + " " + (comment.suggestion or "")).lower()
    if not truth_keywords:
        return True
    overlap = sum(1 for kw in truth_keywords if kw in comment_text)
    return overlap / len(truth_keywords) >= 0.25  # 25% keyword overlap


@dataclass
class MatchResult:
    matched: bool = False
    partial: bool = False  # right category, wrong line
    score: float = 0.0
    reason: str = ""


def _match_comment_to_truth(
    comment: ReviewComment,
    truth_list: List[ReviewComment],
    already_matched: set[int],
) -> tuple[MatchResult, Optional[int]]:
    """Try to match a single agent comment against the ground-truth list."""
    best = MatchResult()
    best_idx: Optional[int] = None

    for idx, truth in enumerate(truth_list):
        if idx in already_matched:
            continue
        if not _category_match(comment, truth):
            continue

        line_ok = _line_close(comment, truth)
        sev_ok = _severity_close(comment, truth)
        msg_ok = _message_relevant(comment, truth)

        if line_ok and msg_ok:
            # Full match
            score = SEVERITY_WEIGHT[truth.severity]
            if sev_ok:
                score *= 1.0
            else:
                score *= 0.7  # severity mismatch penalty
            result = MatchResult(matched=True, partial=False, score=score,
                                 reason=f"TP: {truth.category} L{truth.line}")
            if score > best.score:
                best = result
                best_idx = idx
        elif _category_match(comment, truth) and msg_ok and not line_ok:
            # Partial: right issue, wrong line
            score = SEVERITY_WEIGHT[truth.severity] * 0.5
            result = MatchResult(matched=False, partial=True, score=score,
                                 reason=f"Partial: right issue wrong line for {truth.category}")
            if score > best.score:
                best = result
                best_idx = idx

    return best, best_idx


# ---------------------------------------------------------------------------
# Base grader
# ---------------------------------------------------------------------------

class BaseGrader:
    TASK_ID: str = ""
    CATEGORIES: list[ReviewCategory] = []

    def grade(
        self,
        action: Action,
        ground_truth: List[ReviewComment],
    ) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 1 – Easy (Bug + Style)
# ---------------------------------------------------------------------------

class Task1Grader(BaseGrader):
    TASK_ID = "task_1_easy"
    CATEGORIES = [ReviewCategory.BUG, ReviewCategory.STYLE]

    def grade(self, action: Action, ground_truth: List[ReviewComment]) -> dict:
        comments = action.comments
        matched_truths: set[int] = set()
        tp_score = 0.0
        fp_penalty = 0.0
        breakdown: dict[str, float] = {}

        for comment in comments:
            if comment.category not in self.CATEGORIES:
                fp_penalty += 0.05
                continue
            result, idx = _match_comment_to_truth(comment, ground_truth, matched_truths)
            if result.matched or result.partial:
                tp_score += result.score
                if idx is not None:
                    matched_truths.add(idx)
            else:
                fp_penalty += 0.1  # fabricated issue

        # Max possible TP score
        max_score = sum(SEVERITY_WEIGHT[t.severity] for t in ground_truth
                        if t.category in self.CATEGORIES)
        recall = tp_score / max_score if max_score > 0 else 0.0

        # Penalise missed criticals/highs
        missed_critical_penalty = 0.0
        for idx, truth in enumerate(ground_truth):
            if idx not in matched_truths and truth.severity in (Severity.HIGH, Severity.CRITICAL):
                if truth.category in self.CATEGORIES:
                    missed_critical_penalty += 0.15

        raw = recall - min(fp_penalty, 0.3) - missed_critical_penalty
        final = round(max(0.0, min(1.0, raw)), 4)

        breakdown["recall"] = round(recall, 4)
        breakdown["fp_penalty"] = round(-min(fp_penalty, 0.3), 4)
        breakdown["missed_critical_penalty"] = round(-missed_critical_penalty, 4)

        return {
            "score": final,
            "breakdown": breakdown,
            "matched_count": len(matched_truths),
            "total_ground_truth": len([t for t in ground_truth if t.category in self.CATEGORIES]),
        }


# ---------------------------------------------------------------------------
# Task 2 – Medium (Security + Performance)
# ---------------------------------------------------------------------------

class Task2Grader(BaseGrader):
    TASK_ID = "task_2_medium"
    CATEGORIES = [ReviewCategory.SECURITY, ReviewCategory.PERFORMANCE]

    def grade(self, action: Action, ground_truth: List[ReviewComment]) -> dict:
        comments = action.comments
        matched_truths: set[int] = set()
        tp_score = 0.0
        fp_penalty = 0.0

        for comment in comments:
            if comment.category not in self.CATEGORIES:
                fp_penalty += 0.03
                continue
            result, idx = _match_comment_to_truth(comment, ground_truth, matched_truths)
            if result.matched or result.partial:
                tp_score += result.score
                if idx is not None:
                    matched_truths.add(idx)
            else:
                fp_penalty += 0.12

        max_score = sum(SEVERITY_WEIGHT[t.severity] for t in ground_truth
                        if t.category in self.CATEGORIES)
        recall = tp_score / max_score if max_score > 0 else 0.0

        # Security criticals have double penalty if missed
        missed_penalty = 0.0
        for idx, truth in enumerate(ground_truth):
            if idx not in matched_truths and truth.category == ReviewCategory.SECURITY:
                if truth.severity == Severity.CRITICAL:
                    missed_penalty += 0.20
                elif truth.severity == Severity.HIGH:
                    missed_penalty += 0.10

        raw = recall - min(fp_penalty, 0.3) - missed_penalty
        final = round(max(0.0, min(1.0, raw)), 4)

        return {
            "score": final,
            "breakdown": {
                "recall": round(recall, 4),
                "fp_penalty": round(-min(fp_penalty, 0.3), 4),
                "missed_security_penalty": round(-missed_penalty, 4),
            },
            "matched_count": len(matched_truths),
            "total_ground_truth": len([t for t in ground_truth if t.category in self.CATEGORIES]),
        }


# ---------------------------------------------------------------------------
# Task 3 – Hard (All categories + summary required)
# ---------------------------------------------------------------------------

class Task3Grader(BaseGrader):
    TASK_ID = "task_3_hard"
    CATEGORIES = list(ReviewCategory)

    def grade(self, action: Action, ground_truth: List[ReviewComment]) -> dict:
        comments = action.comments
        matched_truths: set[int] = set()
        tp_score = 0.0
        fp_penalty = 0.0

        for comment in comments:
            result, idx = _match_comment_to_truth(comment, ground_truth, matched_truths)
            if result.matched or result.partial:
                tp_score += result.score
                if idx is not None:
                    matched_truths.add(idx)
            else:
                fp_penalty += 0.08

        max_score = sum(SEVERITY_WEIGHT[t.severity] for t in ground_truth)
        recall = tp_score / max_score if max_score > 0 else 0.0

        # Summary quality bonus (up to +0.15)
        summary_bonus = 0.0
        if action.summary:
            summary_lower = action.summary.lower()
            key_themes = ["security", "injection", "pickle", "performance", "documentation", "bug"]
            hits = sum(1 for kw in key_themes if kw in summary_lower)
            summary_bonus = min(0.15, hits * 0.025)

        # Summary required penalty
        summary_penalty = 0.10 if not action.summary else 0.0

        # Missed critical penalty
        missed_penalty = 0.0
        for idx, truth in enumerate(ground_truth):
            if idx not in matched_truths:
                if truth.severity == Severity.CRITICAL:
                    missed_penalty += 0.15
                elif truth.severity == Severity.HIGH:
                    missed_penalty += 0.08

        raw = recall + summary_bonus - min(fp_penalty, 0.3) - missed_penalty - summary_penalty
        final = round(max(0.0, min(1.0, raw)), 4)

        return {
            "score": final,
            "breakdown": {
                "recall": round(recall, 4),
                "summary_bonus": round(summary_bonus, 4),
                "fp_penalty": round(-min(fp_penalty, 0.3), 4),
                "missed_critical_penalty": round(-missed_penalty, 4),
                "summary_penalty": round(-summary_penalty, 4),
            },
            "matched_count": len(matched_truths),
            "total_ground_truth": len(ground_truth),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADERS: dict[str, BaseGrader] = {
    "task_1_easy": Task1Grader(),
    "task_2_medium": Task2Grader(),
    "task_3_hard": Task3Grader(),
}
