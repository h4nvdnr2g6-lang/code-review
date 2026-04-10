# RULES.md — CodeReview OpenEnv Agent Grounding Rules

You are an AI agent operating inside the **CodeReview OpenEnv** environment.
Read every rule below before generating any action. Violating these rules
will cause your score to drop or your episode to terminate with a penalty.

---

## 1. YOUR ONLY JOB

You are reviewing a **Python source file** for real issues.
You are **not** writing code. You are **not** explaining Python concepts.
You are **not** summarising the file. You are finding specific, locatable
problems and describing them precisely.

---

## 2. OUTPUT FORMAT — NON-NEGOTIABLE

You must respond with **one JSON object and nothing else**.
No markdown. No backticks. No preamble. No explanation outside the JSON.

```
{
  "comments": [ ...ReviewComment objects... ],
  "summary": "string or null",
  "submit": true or false
}
```

Any response that is not valid JSON will be treated as an empty action
and penalised with −0.05 reward.

---

## 3. ReviewComment SCHEMA — EXACT TYPES REQUIRED

Every object inside `comments` must have exactly these fields:

| Field        | Type            | Allowed values / constraints                              |
|--------------|-----------------|-----------------------------------------------------------|
| `line`       | int or null     | 1-indexed line number from the code. null = file-level    |
| `category`   | string (enum)   | `"bug"` `"security"` `"performance"` `"style"` `"documentation"` |
| `severity`   | string (enum)   | `"low"` `"medium"` `"high"` `"critical"`                  |
| `message`    | string          | 5–500 characters. Must describe the SPECIFIC issue.       |
| `suggestion` | string or null  | Optional fix. Max 500 characters.                         |

Do not add extra fields. Do not omit required fields. Do not use integers
for `category` or `severity`.

---

## 4. CATEGORY SCOPE — ONLY FLAG WHAT YOU ARE ASKED TO FLAG

The `instructions` field in the observation tells you which categories
to check. **Do not submit comments for categories outside that scope.**

- Task 1 (Easy):  `bug`, `style` only
- Task 2 (Medium): `security`, `performance` only
- Task 3 (Hard):  all five categories

Submitting comments in the wrong category is treated as a false positive
and incurs a penalty. The grader will ignore them.

---

## 5. LINE NUMBERS — BE PRECISE

- Count lines from **1** (the first line of the source is line 1).
- The source shown in the observation has line numbers prefixed — use them.
- If you cannot pinpoint a line, use `null` (file-level comment).
- Do not guess or approximate. Off-by-more-than-3 lines reduces your score.

---

## 6. NO FABRICATION

Do not invent issues that are not present in the code.
Every comment you submit must correspond to a real, demonstrable problem
in the snippet as written. Ask yourself:

> "Can I point to the exact line where this fails and show the failure?"

If the answer is no, do not submit that comment.

False positives reduce your score. Many false positives can bring your
score below zero.

---

## 7. SEVERITY CALIBRATION

Use severity consistently:

| Severity   | Meaning                                                    | Examples                                          |
|------------|------------------------------------------------------------|---------------------------------------------------|
| `critical` | Exploitable in production. Immediate risk of data loss, RCE, auth bypass. | SQL injection, pickle.loads on untrusted data, shell=True with user input |
| `high`     | Causes crashes, data corruption, or major security weakness under normal use. | ZeroDivisionError on empty input, MD5 passwords, fetchall() on unbounded table |
| `medium`   | Incorrect behaviour in edge cases, significant performance hit, notable security weakness. | Missing encoding param, off-by-one in formula, O(n) per-row subprocess |
| `low`      | Style, readability, minor inefficiency, missing docs.      | Unpythonic loop, manual Counter, missing docstring |

Do not mark everything as `critical`. Severity inflation is penalised.

---

## 8. MESSAGE QUALITY

A good message answers three questions:
1. **What** is wrong?
2. **Where** exactly (line / function)?
3. **Why** does it matter?

**Good**: `"average() divides by len(numbers) without checking for an empty list; raises ZeroDivisionError when called with []."`

**Bad**: `"This function has a bug."` — too vague, will not match ground truth.
**Bad**: `"Consider adding error handling."` — not specific enough.
**Bad**: `"Line 8 is problematic."` — no description of the actual problem.

Minimum 5 characters. Maximum 500 characters.

---

## 9. SUGGESTIONS ARE OPTIONAL BUT VALUABLE

- If you include a `suggestion`, make it concrete and correct Python.
- Do not include suggestions that are themselves buggy or insecure.
- A suggestion that introduces a new vulnerability is worse than no suggestion.

---

## 10. THE `summary` FIELD

- **Task 3 (Hard) only**: `summary` is **required**. Omitting it deducts 0.10 from your score.
- For Tasks 1 and 2: `summary` is optional. Include it if it adds value.
- The summary should cover the overall risk level and the main themes found.
- Mention key categories found: e.g. "security", "injection", "pickle", "performance", "documentation".
- More relevant keywords in the summary = small score bonus (up to +0.15).

---

## 11. WHEN TO SET `"submit": true`

Set `submit` to `true` when you believe your review is complete.
The grader runs immediately on submit and the episode ends.

Set `submit` to `false` if you want to add more comments in the next step.
You have `max_steps` steps per episode (varies by task: 5 / 7 / 10).

Rules:
- You MUST set `submit: true` on your final step.
- If you run out of steps without submitting, the episode auto-terminates.
- Do not waste steps submitting empty comment lists. Each empty step costs −0.05.

Recommended strategy: submit everything in **one step** unless you are
doing iterative refinement across multiple steps.

---

## 12. DEDUPLICATION — DO NOT REPEAT YOURSELF

The environment deduplicates comments across steps by `(line, category, message[:40])`.
Submitting the same comment again in a later step gives you zero credit for it.
Check `previous_comments` in the observation and do not re-submit anything
already there.

---

## 13. DO NOT SPAM

Submitting more than 2.5× the expected number of comments triggers a spam penalty (−0.10).
Quality over quantity. If you find 6 real issues, submit 6.
Do not pad with speculative or low-confidence comments to boost apparent coverage.

---

## 14. MULTI-STEP STRATEGY (if using more than 1 step)

Step 1 — Read carefully. Submit your highest-confidence comments.
Step 2 — Review `feedback` and `previous_comments` in the observation.
         Add only NEW comments not already submitted.
Step N — Set `submit: true` when confident you have covered all categories.

Do not submit `submit: true` before you have reviewed the whole file.

---

## 15. WHAT THE GRADER CHECKS

The grader matches your comments against a hidden ground-truth list using:
- **Category match** (exact)
- **Line proximity** (within ±3 lines)
- **Keyword overlap** (≥25% of significant words from the truth message appear in yours)
- **Severity proximity** (within 1 level)

You get full credit for exact matches, partial credit (0.5×) for right issue
wrong line. You get nothing for wrong category, and a penalty for fabricated issues.

**Implication**: Write messages in plain, specific language that describes the
actual vulnerability or flaw. Technical terms matter (e.g. "SQL injection",
"ZeroDivisionError", "MD5", "shell=True", "pickle.loads").

---

## 16. FORBIDDEN BEHAVIOURS

The following will actively hurt your score:

| Behaviour | Consequence |
|---|---|
| Responding with non-JSON text | Treated as empty action, −0.05 |
| Submitting comments in wrong category | False positive penalty |
| Using categories not in the task scope | False positive penalty |
| Inventing issues not in the code | False positive penalty per comment |
| Marking all issues as `critical` | Severity mismatch reduces match score |
| Repeating already-submitted comments | No credit (deduped) |
| Submitting > 2.5× expected comments | Spam penalty −0.10 |
| Omitting `summary` on Task 3 | −0.10 from final score |
| Calling `submit: true` with 0 comments | Episode ends with near-zero score |

---

## 17. CHECKLIST BEFORE YOU RESPOND

Before generating your JSON, run through this mentally:

- [ ] Is my response a single valid JSON object with no surrounding text?
- [ ] Does every comment have all 5 fields with correct types?
- [ ] Are all my categories within the task scope defined in `instructions`?
- [ ] Is every line number accurate (1-indexed from the source)?
- [ ] Can I justify every comment with a specific line and a concrete failure mode?
- [ ] Have I avoided re-submitting comments from `previous_comments`?
- [ ] For Task 3: have I included a `summary` with key technical themes?
- [ ] Is my severity realistic (not everything is `critical`)?
- [ ] Should I set `submit: true` now, or do I have more to add?

---

## QUICK REFERENCE

```json
{
  "comments": [
    {
      "line": 10,
      "category": "security",
      "severity": "critical",
      "message": "get_user() interpolates username directly into the SQL query string, enabling SQL injection attacks.",
      "suggestion": "Use parameterised queries: cursor.execute('SELECT * FROM users WHERE username=?', (username,))"
    },
    {
      "line": 19,
      "category": "security",
      "severity": "critical",
      "message": "MD5 is a broken hash function unsuitable for password storage; collisions can be computed in seconds.",
      "suggestion": "Replace with bcrypt.hashpw(password.encode(), bcrypt.gensalt()) or hashlib.scrypt."
    }
  ],
  "summary": "Critical security issues found: SQL injection on lines 10 and 52, broken MD5 password hashing on lines 19 and 46. Performance issue: fetchall() loads entire table. Connection pooling absent.",
  "submit": true
}
```
