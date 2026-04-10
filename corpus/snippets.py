"""
Code corpus: Python snippets with embedded ground-truth issues.

Each entry has:
  - snippet  : CodeSnippet to show the agent
  - issues   : list of ground-truth ReviewComment objects the grader checks against
  - task_id  : which task this belongs to
"""

from __future__ import annotations

from env.models import CodeSnippet, ReviewCategory, ReviewComment, Severity

# ---------------------------------------------------------------------------
# TASK 1 – Easy  (Bug detection + Code style)
# ---------------------------------------------------------------------------

TASK1_SNIPPET = CodeSnippet(
    file_name="calculator.py",
    source='''\
def divide(a, b):
    return a / b                          # line 2

def average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)           # line 8

def celsius_to_fahrenheit(c):
    return c * 9/5 + 31                  # line 11  (bug: should be +32)

def is_palindrome(s):
    return s == s[::-1]                   # line 14

def find_max(lst):
    max_val = lst[0]                      # line 17
    for i in range(len(lst)):
        if lst[i] > max_val:
            max_val = lst[i]
    return max_val                        # line 21

def count_words(text):
    words = text.split(" ")
    wordcount = {}
    for w in words:
        if w in wordcount:
            wordcount[w] = wordcount[w]+1
        else:
            wordcount[w] = 1
    return wordcount                      # line 30
''',
)

TASK1_ISSUES: list[ReviewComment] = [
    # ---- Bugs ----
    ReviewComment(
        line=2,
        category=ReviewCategory.BUG,
        severity=Severity.HIGH,
        message="divide() has no guard against division by zero; will raise ZeroDivisionError when b=0.",
        suggestion="Add `if b == 0: raise ValueError('b must not be zero')` before returning.",
    ),
    ReviewComment(
        line=8,
        category=ReviewCategory.BUG,
        severity=Severity.HIGH,
        message="average() crashes with ZeroDivisionError on an empty list.",
        suggestion="Guard with `if not numbers: return 0.0` or raise ValueError.",
    ),
    ReviewComment(
        line=11,
        category=ReviewCategory.BUG,
        severity=Severity.MEDIUM,
        message="celsius_to_fahrenheit uses +31 instead of +32, giving wrong results.",
        suggestion="Change `+ 31` to `+ 32`.",
    ),
    ReviewComment(
        line=17,
        category=ReviewCategory.BUG,
        severity=Severity.MEDIUM,
        message="find_max() crashes with IndexError on an empty list.",
        suggestion="Add `if not lst: raise ValueError('list is empty')` at the top.",
    ),
    # ---- Style ----
    ReviewComment(
        line=18,
        category=ReviewCategory.STYLE,
        severity=Severity.LOW,
        message="Iterating with `for i in range(len(lst))` is unpythonic; prefer `for val in lst`.",
        suggestion="Replace loop body with `for val in lst: if val > max_val: max_val = val`.",
    ),
    ReviewComment(
        line=25,
        category=ReviewCategory.STYLE,
        severity=Severity.LOW,
        message="count_words manually reimplements collections.Counter; use the stdlib instead.",
        suggestion="Replace with `from collections import Counter; return Counter(text.split())`.",
    ),
]

# ---------------------------------------------------------------------------
# TASK 2 – Medium  (Security + Performance)
# ---------------------------------------------------------------------------

TASK2_SNIPPET = CodeSnippet(
    file_name="user_service.py",
    source='''\
import sqlite3
import hashlib
import os

DB_PATH = "users.db"

def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = \'{ username }\'"   # line 10
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

def create_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    pw_hash = hashlib.md5(password.encode()).hexdigest()               # line 19
    cursor.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, pw_hash),
    )
    conn.commit()
    conn.close()

def load_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()                                           # line 31
    conn.close()
    users = []
    for row in rows:
        users.append({
            "id": row[0],
            "username": row[1],
            "password": row[2],
        })
    return users

def authenticate(username, password):
    user = get_user(username)
    if user is None:
        return False
    pw_hash = hashlib.md5(password.encode()).hexdigest()               # line 46
    return user[2] == pw_hash

def delete_user(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = f"DELETE FROM users WHERE username = \'{ username }\'"    # line 52
    cursor.execute(query)
    conn.commit()
    conn.close()
''',
)

TASK2_ISSUES: list[ReviewComment] = [
    # ---- Security ----
    ReviewComment(
        line=10,
        category=ReviewCategory.SECURITY,
        severity=Severity.CRITICAL,
        message="SQL injection vulnerability: username is interpolated directly into the query string.",
        suggestion="Use parameterised queries: `cursor.execute('SELECT * FROM users WHERE username=?', (username,))`",
    ),
    ReviewComment(
        line=19,
        category=ReviewCategory.SECURITY,
        severity=Severity.CRITICAL,
        message="MD5 is cryptographically broken and must not be used for password hashing.",
        suggestion="Replace with `bcrypt.hashpw(password.encode(), bcrypt.gensalt())` or `hashlib.scrypt`.",
    ),
    ReviewComment(
        line=52,
        category=ReviewCategory.SECURITY,
        severity=Severity.CRITICAL,
        message="delete_user() is also vulnerable to SQL injection via string interpolation.",
        suggestion="Use parameterised queries: `cursor.execute('DELETE FROM users WHERE username=?', (username,))`",
    ),
    ReviewComment(
        line=46,
        category=ReviewCategory.SECURITY,
        severity=Severity.HIGH,
        message="authenticate() re-hashes with MD5 for comparison; same broken-hash issue as create_user.",
        suggestion="Adopt bcrypt.checkpw() or equivalent constant-time comparison.",
    ),
    # ---- Performance ----
    ReviewComment(
        line=31,
        category=ReviewCategory.PERFORMANCE,
        severity=Severity.HIGH,
        message="fetchall() loads the entire users table into memory; will OOM on large tables.",
        suggestion="Use `cursor.fetchmany(size=1000)` in a loop or add a LIMIT clause.",
    ),
    ReviewComment(
        line=8,
        category=ReviewCategory.PERFORMANCE,
        severity=Severity.MEDIUM,
        message="A new DB connection is opened and closed for every single query; connection pooling should be used.",
        suggestion="Use a module-level connection or a context-manager pool (e.g. `sqlite3.connect` as a shared resource).",
    ),
]

# ---------------------------------------------------------------------------
# TASK 3 – Hard  (All categories: Bug + Security + Performance + Style + Docs)
# ---------------------------------------------------------------------------

TASK3_SNIPPET = CodeSnippet(
    file_name="data_pipeline.py",
    source='''\
"""Data pipeline for processing CSV exports from the analytics platform."""

import csv
import os
import pickle
import subprocess
import time

CACHE = {}

def load_data(filepath):
    with open(filepath) as f:                                         # line 12
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            data.append(row)
    return data

def process_row(row, transform_script):
    result = subprocess.run(transform_script, shell=True, input=str(row))  # line 20
    return result.stdout

def cache_result(key, value):
    CACHE[key] = pickle.dumps(value)                                  # line 24

def get_cached(key):
    if key in CACHE:
        return pickle.loads(CACHE[key])                               # line 28

def compute_stats(data):
    n = len(data)                                                     # line 31
    total = sum(float(row["value"]) for row in data)
    mean = total / n
    variance = sum((float(row["value"]) - mean) ** 2 for row in data) / n
    return {"mean": mean, "variance": variance, "count": n}

def run_pipeline(filepath, transform_script=None):
    data = load_data(filepath)
    if transform_script:
        processed = []
        for row in data:
            processed.append(process_row(row, transform_script))
        data = processed
    stats = compute_stats(data)
    cache_result(filepath, stats)
    return stats

def export_results(stats, output_path):
    with open(output_path, "w") as f:                                 # line 47
        f.write(str(stats))
''',
)

TASK3_ISSUES: list[ReviewComment] = [
    # ---- Security ----
    ReviewComment(
        line=20,
        category=ReviewCategory.SECURITY,
        severity=Severity.CRITICAL,
        message="subprocess.run with shell=True and user-supplied transform_script enables arbitrary OS command injection.",
        suggestion="Avoid shell=True; pass args as a list or whitelist allowed scripts.",
    ),
    ReviewComment(
        line=28,
        category=ReviewCategory.SECURITY,
        severity=Severity.CRITICAL,
        message="pickle.loads() on untrusted/arbitrary cache data allows arbitrary code execution.",
        suggestion="Replace pickle with json.dumps/loads for serialisable data, or sign+verify the payload.",
    ),
    ReviewComment(
        line=24,
        category=ReviewCategory.SECURITY,
        severity=Severity.HIGH,
        message="Storing pickled data in a module-level dict means deserialization risk persists across calls.",
        suggestion="Use JSON for the cache and validate schemas on retrieval.",
    ),
    # ---- Bugs ----
    ReviewComment(
        line=31,
        category=ReviewCategory.BUG,
        severity=Severity.HIGH,
        message="compute_stats() raises ZeroDivisionError when data is empty (n=0).",
        suggestion="Guard with `if not data: return {'mean': 0, 'variance': 0, 'count': 0}`.",
    ),
    ReviewComment(
        line=32,
        category=ReviewCategory.BUG,
        severity=Severity.MEDIUM,
        message="If any row is missing the 'value' key, a KeyError will silently abort the pipeline.",
        suggestion="Use `row.get('value', 0)` or validate schema at load time.",
    ),
    ReviewComment(
        line=12,
        category=ReviewCategory.BUG,
        severity=Severity.MEDIUM,
        message="open(filepath) without encoding='utf-8' will use the system locale; may fail on non-ASCII data.",
        suggestion="Use `open(filepath, encoding='utf-8')`.",
    ),
    # ---- Performance ----
    ReviewComment(
        line=31,
        category=ReviewCategory.PERFORMANCE,
        severity=Severity.MEDIUM,
        message="compute_stats() iterates over data twice (once for sum, once for variance); single-pass Welford's algorithm is more efficient.",
        suggestion="Use Welford's online algorithm or numpy for large datasets.",
    ),
    ReviewComment(
        line=38,
        category=ReviewCategory.PERFORMANCE,
        severity=Severity.MEDIUM,
        message="process_row() spawns a new subprocess for every row; should batch or vectorise the transformation.",
        suggestion="Pass all rows to a single subprocess call or use a Python-native transform function.",
    ),
    # ---- Style ----
    ReviewComment(
        line=47,
        category=ReviewCategory.STYLE,
        severity=Severity.LOW,
        message="export_results writes str(stats) (a Python dict repr) rather than valid JSON or CSV.",
        suggestion="Use `import json; f.write(json.dumps(stats, indent=2))`.",
    ),
    ReviewComment(
        line=9,
        category=ReviewCategory.STYLE,
        severity=Severity.LOW,
        message="Module-level mutable CACHE dict is a global side-effect; makes the pipeline hard to test and thread-unsafe.",
        suggestion="Encapsulate state inside a Pipeline class or pass cache explicitly.",
    ),
    # ---- Documentation ----
    ReviewComment(
        line=12,
        category=ReviewCategory.DOCUMENTATION,
        severity=Severity.LOW,
        message="load_data() has no docstring; expected CSV schema (required columns, types) is undocumented.",
        suggestion="Add a docstring describing filepath, expected columns, and return type.",
    ),
    ReviewComment(
        line=19,
        category=ReviewCategory.DOCUMENTATION,
        severity=Severity.LOW,
        message="process_row() does not document what transform_script should be, its expected format, or return value.",
        suggestion="Add docstring: args, expected script interface, return type, and example.",
    ),
    ReviewComment(
        line=None,
        category=ReviewCategory.DOCUMENTATION,
        severity=Severity.LOW,
        message="Module-level docstring is too vague; doesn't mention side-effects, required CSV schema, or dependencies.",
        suggestion="Expand the module docstring with usage example, required columns, and external dependencies.",
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CORPUS: dict[str, dict] = {
    "task_1_easy": {
        "snippet": TASK1_SNIPPET,
        "issues": TASK1_ISSUES,
    },
    "task_2_medium": {
        "snippet": TASK2_SNIPPET,
        "issues": TASK2_ISSUES,
    },
    "task_3_hard": {
        "snippet": TASK3_SNIPPET,
        "issues": TASK3_ISSUES,
    },
}
