"""Tool: execute LLM-generated Python code — ALWAYS asks the user for confirmation first.

This is the 'generated code' path from PROJECT.md: predefined tools run freely,
but ad-hoc code written by the LLM is shown to the user and only executed after
an explicit yes.
"""

import os
import subprocess
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT_S = 30
MAX_OUTPUT = 2000

TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute a short Python snippet in the workspace. Use ONLY when no other tool can do the job. The user sees the code and must confirm before it runs.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute. Keep it short and print its results.",
                }
            },
            "required": ["code"],
        },
    },
}


def run(code):
    print("\n" + "─" * 60)
    print("⚠️  The assistant wants to run this Python code:")
    print("─" * 60)
    print(code)
    print("─" * 60)
    try:
        answer = input("Execute? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        return "The user DECLINED to run this code. Do not retry; tell the user it was not executed."

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            cwd=WORKSPACE,
        )
    except subprocess.TimeoutExpired:
        return f"error: execution timed out after {TIMEOUT_S}s"
    output = (proc.stdout + proc.stderr).strip()
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + "\n... (truncated)"
    return f"exit code {proc.returncode}\n{output or '(no output)'}"
