#!/usr/bin/env python3
"""
baseline_agent.py – Baseline inference script for CodeReview OpenEnv.

Runs gpt-4o against all three tasks using the OpenAI client.
Reads credentials from OPENAI_API_KEY environment variable.
Connects to the env either locally (direct Python import) or via HTTP.

Usage
-----
    # Direct mode (no server needed):
    python baseline_agent.py

    # Against a running server:
    python baseline_agent.py --mode http --base-url http://localhost:7860

    # Single task:
    python baseline_agent.py --task task_2_medium
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.environ.get("BASELINE_MODEL", "gpt-4o")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]

# ---------------------------------------------------------------------------
# Prompt construction
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


def build_user_message(observation: dict) -> str:
    snippet = observation["snippet"]
    instructions = observation["instructions"]
    previous = observation.get("previous_comments", [])

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
            msg += f"  - L{c.get('line','?')} [{c['category']}] {c['message'][:80]}\n"

    return msg.strip()


# ---------------------------------------------------------------------------
# Direct mode (import env directly)
# ---------------------------------------------------------------------------

def run_direct(task_id: str, client: OpenAI) -> dict:
    """Run the agent against the environment by direct Python import."""
    # Import here to avoid circular dependency when running in HTTP mode
    sys.path.insert(0, os.path.dirname(__file__))
    from env.environment import CodeReviewEnv
    from env.models import Action, ReviewComment, ReviewCategory, Severity

    env = CodeReviewEnv(task_id=task_id)
    obs = env.reset()

    total_reward = 0.0
    final_score = 0.0
    steps_taken = 0

    for step_num in range(env.spec.max_steps):
        user_msg = build_user_message(obs.model_dump())

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            action_dict = json.loads(raw)
        except Exception as e:
            print(f"  [!] LLM error on step {step_num}: {e}")
            action_dict = {"comments": [], "submit": True}

        # Build Action
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
            submit=action_dict.get("submit", True),
        )

        result = env.step(action)
        total_reward += result.reward.value
        steps_taken += 1
        final_score = result.info.get("grader", {}).get("score", 0.0)

        print(f"  Step {step_num+1}: reward={result.reward.value:+.3f} | "
              f"comments={result.info['total_comments']} | "
              f"score={final_score:.3f}")

        obs = result.observation
        if result.done:
            break

    passed = final_score >= env.spec.passing_threshold
    return {
        "task_id": task_id,
        "steps": steps_taken,
        "total_reward": round(total_reward, 4),
        "final_score": round(final_score, 4),
        "passed": passed,
        "threshold": env.spec.passing_threshold,
    }


# ---------------------------------------------------------------------------
# HTTP mode (against a running server)
# ---------------------------------------------------------------------------

def run_http(task_id: str, client: OpenAI, base_url: str) -> dict:
    """Run the agent against a live HTTP server."""
    session_id = f"baseline-{task_id}-{int(time.time())}"
    headers = {"Content-Type": "application/json"}

    # Reset
    r = requests.post(f"{base_url}/reset",
                      json={"task_id": task_id, "session_id": session_id}, headers=headers)
    r.raise_for_status()
    obs = r.json()["observation"]

    # Get task spec for threshold
    tasks_r = requests.get(f"{base_url}/tasks")
    spec = tasks_r.json()[task_id]
    max_steps = spec["max_steps"]
    threshold = spec["passing_threshold"]

    total_reward = 0.0
    final_score = 0.0
    steps_taken = 0

    for step_num in range(max_steps):
        user_msg = build_user_message(obs)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            action_dict = json.loads(response.choices[0].message.content or "{}")
        except Exception as e:
            print(f"  [!] LLM error: {e}")
            action_dict = {"comments": [], "submit": True}

        step_r = requests.post(
            f"{base_url}/step",
            json={"session_id": session_id, "action": action_dict},
            headers=headers,
        )
        step_r.raise_for_status()
        result = step_r.json()

        total_reward += result["reward"]["value"]
        steps_taken += 1
        final_score = result["info"].get("grader", {}).get("score", 0.0)

        print(f"  Step {step_num+1}: reward={result['reward']['value']:+.3f} | "
              f"comments={result['info']['total_comments']} | "
              f"score={final_score:.3f}")

        obs = result["observation"]
        if result["done"]:
            break

    return {
        "task_id": task_id,
        "steps": steps_taken,
        "total_reward": round(total_reward, 4),
        "final_score": round(final_score, 4),
        "passed": final_score >= threshold,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline agent for CodeReview OpenEnv")
    parser.add_argument("--mode", choices=["direct", "http"], default="direct")
    parser.add_argument("--base-url", default=ENV_BASE_URL)
    parser.add_argument("--task", choices=TASKS + ["all"], default="all")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY)
    tasks_to_run = TASKS if args.task == "all" else [args.task]

    print(f"\n{'='*60}")
    print(f"  CodeReview OpenEnv – Baseline Agent ({MODEL})")
    print(f"  Mode: {args.mode}")
    print(f"{'='*60}\n")

    results: List[dict] = []
    for task_id in tasks_to_run:
        print(f"▶ Running {task_id} ...")
        t0 = time.time()
        if args.mode == "direct":
            r = run_direct(task_id, client)
        else:
            r = run_http(task_id, client, args.base_url)
        elapsed = round(time.time() - t0, 1)
        r["elapsed_s"] = elapsed
        results.append(r)
        status = "✅ PASSED" if r["passed"] else "❌ FAILED"
        print(f"  → {status} | score={r['final_score']:.3f} | reward={r['total_reward']:+.3f} | {elapsed}s\n")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"  {'Task':<22} {'Score':>7} {'Threshold':>10} {'Reward':>8} {'Pass':>6}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['task_id']:<22} {r['final_score']:>7.3f} {r['threshold']:>10.2f} "
              f"{r['total_reward']:>+8.3f} {'✅' if r['passed'] else '❌':>6}")
    avg_score = sum(r["final_score"] for r in results) / len(results)
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    print(f"  {'-'*55}")
    print(f"  {'AVERAGE':<22} {avg_score:>7.3f} {'':>10} {'':>8} {pass_rate*100:>5.0f}%")
    print(f"{'='*60}\n")

    # Save results
    out_path = "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "results": results}, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
