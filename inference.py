#!/usr/bin/env python3
"""
Inference Script for CodeReview OpenEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after the episode, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=task_1_easy env=code_review model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=review(comments=6,submit=true) reward=0.85 done=true error=null
    [END] success=true steps=1 score=0.850 rewards=0.85
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Ensure project root is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import CodeReviewEnv, TASK_SPECS
from env.models import Action, ReviewComment, ReviewCategory, Severity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # If using docker image
HF_TOKEN = os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("BENCHMARK", "code_review")
TASK_NAME = os.getenv("CODE_REVIEW_TASK", "all")  # "all" or a specific task id
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]
TEMPERATURE = 0.2
MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# System prompt for code review
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Python code reviewer.
You will be given a code snippet along with review instructions.
Your job is to produce a JSON action object that identifies issues in the code.

The JSON object you return must match this schema exactly:
{
  "comments": [
    {
      "line": <int or null>,
      "category": <"bug"|"security"|"performance"|"style"|"documentation">,
      "severity": <"low"|"medium"|"high"|"critical">,
      "message": "<clear description of the issue>",
      "suggestion": "<optional fix>"
    }
  ],
  "summary": "<overall assessment – required for hard tasks, optional otherwise>",
  "submit": true
}

Rules:
- Only flag genuine issues. Do not fabricate problems.
- Be precise about line numbers (1-indexed from the code).
- Match the categories listed in the instructions.
- Always set "submit": true when you believe your review is complete.
- Return ONLY the JSON object. No markdown, no explanations.
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers  (exact STDOUT format from spec)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def build_user_message(obs_dict: dict) -> str:
    """Build LLM user prompt from an observation dict."""
    snippet = obs_dict["snippet"]
    instructions = obs_dict["instructions"]
    previous = obs_dict.get("previous_comments", [])

    numbered_source = "\n".join(
        f"{i+1:3d}  {line}"
        for i, line in enumerate(snippet["source"].splitlines())
    )

    msg = f"""
{instructions}

### File: {snippet['file_name']}
```python
{numbered_source}
```
"""
    if previous:
        msg += f"\n### Your previous comments ({len(previous)} so far):\n"
        for c in previous:
            line_val = c.get("line", "?")
            category = c.get("category", "?")
            message = c.get("message", "")[:80]
            msg += f"  - L{line_val} [{category}] {message}\n"

    return msg.strip()


def get_model_action(client: OpenAI, obs_dict: dict) -> dict:
    """Call the LLM and return a parsed action dict."""
    user_msg = build_user_message(obs_dict)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=False,
        )
        raw = (completion.choices[0].message.content or "{}").strip()
        action_dict = json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        action_dict = {"comments": [], "submit": True}

    return action_dict


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(action_dict: dict) -> Action:
    """Convert a raw action dict into a typed Action model."""
    comments: List[ReviewComment] = []
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

    return Action(
        comments=comments,
        summary=action_dict.get("summary"),
        submit=action_dict.get("submit", True),
    )


def format_action_str(action_dict: dict) -> str:
    """Format action dict into a compact string for STEP logging."""
    n = len(action_dict.get("comments", []))
    submit = str(action_dict.get("submit", False)).lower()
    return f"review(comments={n},submit={submit})"


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

async def run_task(task_id: str, client: OpenAI) -> dict:
    """Run a single code-review task episode and return results."""
    env = CodeReviewEnv(task_id=task_id)
    obs = env.reset()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, env.spec.max_steps + 1):
            obs_dict = obs.model_dump()

            # Get LLM response
            action_dict = get_model_action(client, obs_dict)
            action = parse_action(action_dict)

            # Step the environment
            result = env.step(action)

            reward = result.reward.value
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            action_str = format_action_str(action_dict)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            obs = result.observation

            if done:
                score = result.info.get("grader", {}).get("score", 0.0)
                success = score >= env.spec.passing_threshold
                break

    except Exception as e:
        print(f"[DEBUG] Error during task {task_id}: {e}", file=sys.stderr, flush=True)

    finally:
        # Clamp score to [0, 1]
        score = min(max(score, 0.0), 1.0)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks_to_run = TASKS if TASK_NAME == "all" else [TASK_NAME]

    results: List[dict] = []
    for task_id in tasks_to_run:
        result = await run_task(task_id, client)
        results.append(result)

    # Print final summary to stderr (not part of the spec, but useful for debugging)
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    pass_count = sum(1 for r in results if r["success"])
    print(
        f"\n[SUMMARY] tasks={len(results)} passed={pass_count} avg_score={avg_score:.3f}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
