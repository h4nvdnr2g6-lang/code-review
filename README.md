---
title: CodeReview OpenEnv
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 🔍 CodeReview OpenEnv

An **OpenEnv-compliant AI training environment** that simulates professional Python code review. Agents learn to identify bugs, security vulnerabilities, performance bottlenecks, style issues, and documentation gaps — exactly as a senior engineer would in a real pull-request workflow.

---

## Why Code Review?

Code review is one of the highest-leverage tasks in software engineering. It is:

- **Real-world**: Every professional software team does it daily
- **Structured enough to grade**: Issues have objectively correct or incorrect assessments
- **Rich in partial signal**: An agent that spots 3/5 critical issues is measurably better than one that spots 1/5
- **Scalable in difficulty**: Easy (bugs only) → Hard (all categories + written summary)

This makes it an ideal domain for training and evaluating LLM-based agents on multi-step reasoning and quality estimation tasks.

---

## Environment Description

```
CodeReviewEnv
├── Task 1 – Easy    : Bug detection + Code style        (calculator.py, 31 lines)
├── Task 2 – Medium  : Security + Performance audit      (user_service.py, 55 lines)
└── Task 3 – Hard    : Full review, all 5 categories     (data_pipeline.py, 49 lines)
```

Each task presents a Python snippet containing intentional flaws. The agent submits `ReviewComment` objects across one or more steps, then finalises with `submit=True`. A deterministic grader scores the review against ground-truth issues.

---

## Observation Space

What the agent sees on each step:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Active task identifier |
| `step` | `int` | Current step (0-indexed) |
| `snippet.file_name` | `str` | Logical file name (e.g. `auth.py`) |
| `snippet.source` | `str` | Full Python source code |
| `instructions` | `str` | Review scope, difficulty, and guidance |
| `previous_comments` | `list[ReviewComment]` | All comments submitted so far |
| `feedback` | `str \| None` | Env feedback on the last action |
| `done` | `bool` | Whether the episode has ended |

---

## Action Space

What the agent submits on each step:

```json
{
  "comments": [
    {
      "line": 10,
      "category": "security",
      "severity": "critical",
      "message": "SQL injection via string interpolation in query.",
      "suggestion": "Use parameterised queries: cursor.execute('...', (username,))"
    }
  ],
  "summary": "Overall review summary (required for task_3_hard)",
  "submit": true
}
```

| Field | Type | Values |
|---|---|---|
| `comments[].line` | `int \| null` | 1-indexed line number; `null` for file-level |
| `comments[].category` | `enum` | `bug`, `security`, `performance`, `style`, `documentation` |
| `comments[].severity` | `enum` | `low`, `medium`, `high`, `critical` |
| `comments[].message` | `str` | 5–500 chars |
| `comments[].suggestion` | `str \| null` | Optional fix suggestion |
| `summary` | `str \| null` | Required for `task_3_hard`, optional otherwise |
| `submit` | `bool` | `true` finalises the review and triggers the grader |

---

## Reward Function

Rewards are shaped to provide signal over the **full trajectory**, not just on terminal submit.

### Per-step (incremental) rewards

| Event | Reward |
|---|---|
| New valid comment added | `+0.05` per comment (max `+0.15`) |
| Progress signal (grader score delta) | `+0.5 × Δscore` |
| Empty step (no new comments) | `−0.05` |
| Spam (> 2.5× expected comments) | `−0.10` |

### On `submit=True` (terminal)

```
submit_reward = score × 0.8 + (0.2 if score ≥ threshold else −0.2)
```

### Per-category penalties (applied to terminal grader score)

| Event | Penalty |
|---|---|
| False positive (fabricated issue) | `−0.08–0.12` per comment |
| Missed CRITICAL security issue | `−0.15–0.20` |
| Missed HIGH issue | `−0.08–0.10` |
| No summary on task 3 | `−0.10` |

All rewards are clipped to `[−1.0, 1.0]`.

---

## Task Descriptions

### Task 1 – Easy: Bug Detection & Style Review
**File**: `calculator.py` (31 lines) | **Max steps**: 5 | **Pass threshold**: 0.55

Covers basic utility functions: `divide`, `average`, `celsius_to_fahrenheit`, `find_max`, `count_words`.

**Ground-truth issues (6)**:
- `divide()` — no zero-division guard (HIGH bug)
- `average()` — crashes on empty list (HIGH bug)
- `celsius_to_fahrenheit` — off-by-one (+31 vs +32) (MEDIUM bug)
- `find_max()` — crashes on empty list (MEDIUM bug)
- `for i in range(len(lst))` — unpythonic iteration (LOW style)
- Manual `Counter` reimplementation (LOW style)

---

### Task 2 – Medium: Security & Performance Audit
**File**: `user_service.py` (55 lines) | **Max steps**: 7 | **Pass threshold**: 0.60

A SQLite-backed user management service with authentication.

**Ground-truth issues (6)**:
- SQL injection in `get_user()` — f-string query (CRITICAL security)
- MD5 password hashing in `create_user()` (CRITICAL security)
- SQL injection in `delete_user()` (CRITICAL security)
- MD5 reuse in `authenticate()` (HIGH security)
- `fetchall()` on unbounded table (HIGH performance)
- New DB connection per query, no pooling (MEDIUM performance)

---

### Task 3 – Hard: Comprehensive Code Review
**File**: `data_pipeline.py` (49 lines) | **Max steps**: 10 | **Pass threshold**: 0.65

An analytics data pipeline with CSV loading, row transformation, caching, and stats.

**Ground-truth issues (13 across all 5 categories)**:
- `subprocess.run(shell=True)` with user input — OS command injection (CRITICAL security)
- `pickle.loads()` on arbitrary cache data — RCE risk (CRITICAL security)
- Pickling into module-level dict (HIGH security)
- `compute_stats()` ZeroDivisionError on empty data (HIGH bug)
- Missing `"value"` key → silent KeyError (MEDIUM bug)
- `open()` without encoding (MEDIUM bug)
- Two-pass iteration in `compute_stats` (MEDIUM performance)
- Subprocess per row instead of batching (MEDIUM performance)
- `str(stats)` instead of JSON export (LOW style)
- Module-level mutable global cache (LOW style)
- `load_data()` missing docstring (LOW documentation)
- `process_row()` missing docstring (LOW documentation)
- Insufficient module-level docstring (LOW documentation)

A **written summary** is required (`summary` field) — absence incurs a `−0.10` score penalty.

---

## Expected Baseline Scores (gpt-4o)

| Task | Score | Pass? | Notes |
|---|---|---|---|
| `task_1_easy` | ~0.75 | ✅ | GPT-4o reliably spots ZeroDivisionError and off-by-one |
| `task_2_medium` | ~0.65 | ✅ | SQL injection found; MD5 usually flagged; perf issues partial |
| `task_3_hard` | ~0.55 | ✅ | Pickle RCE and shell injection found; docs often missed |

---

## Setup & Usage

### Option A — Docker (recommended)

```bash
# Build
docker build -t code-review-env .

# Run (port 7860)
docker run -p 7860:7860 code-review-env

# Test it
curl http://localhost:7860/health
```

### Option B — Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Open docs
open http://localhost:7860/docs
```

### Run the test suite

```bash
pytest tests/ -v
# Expected: 25 passed
```

### Run the baseline agent

```bash
export OPENAI_API_KEY=sk-...

# All tasks (direct mode — no server needed)
python baseline_agent.py

# Single task
python baseline_agent.py --task task_2_medium

# Against a running HTTP server
python baseline_agent.py --mode http --base-url http://localhost:7860
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | HTML landing page |
| `/health` | GET | Health check |
| `/tasks` | GET | List all task specs |
| `/reset` | POST | Start or restart an episode |
| `/step` | POST | Submit an action |
| `/state` | GET | Get full serialisable state |
| `/docs` | GET | Interactive Swagger UI |

### Example: Full episode via curl

```bash
# 1. Reset
curl -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_1_easy", "session_id": "demo"}'

# 2. Step
curl -X POST http://localhost:7860/step \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "action": {
      "comments": [
        {
          "line": 2,
          "category": "bug",
          "severity": "high",
          "message": "divide() will raise ZeroDivisionError when b is 0.",
          "suggestion": "Guard with: if b == 0: raise ValueError"
        }
      ],
      "submit": true
    }
  }'

# 3. Check state
curl "http://localhost:7860/state?session_id=demo"
```

---

## Project Structure

```
openenv-code-review/
├── app.py                  # FastAPI HTTP server
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Container definition
├── requirements.txt
├── baseline_agent.py       # gpt-4o baseline inference script
│
├── env/
│   ├── models.py           # Pydantic typed models (Observation, Action, Reward, …)
│   └── environment.py      # CodeReviewEnv — step() / reset() / state()
│
├── corpus/
│   └── snippets.py         # Python snippets with ground-truth issues
│
├── graders/
│   └── graders.py          # Task1Grader, Task2Grader, Task3Grader
│
└── tests/
    └── test_env.py         # 25-test pytest suite (all passing)
```

---

## Deploying to Hugging Face Spaces

1. Create a new Space with **Docker** SDK
2. Push this repository to the Space
3. Set `OPENAI_API_KEY` as a Space secret (only needed for baseline script)
4. The Space will auto-build and expose port 7860

```yaml
# README.md frontmatter for HF Spaces
---
title: CodeReview OpenEnv
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - code-review
  - ai-agent
  - evaluation
---
```

---

## License

MIT
