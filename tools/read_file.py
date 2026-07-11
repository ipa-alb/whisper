"""Tool: read a text file from the workspace."""

import os

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_CHARS = 8000

TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file from the workspace and return its content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace root, e.g. 'README.md' or 'notes/todo.md'.",
                }
            },
            "required": ["path"],
        },
    },
}


def run(path):
    full = os.path.realpath(os.path.join(WORKSPACE, path))
    if not full.startswith(WORKSPACE + os.sep):
        return "error: path escapes the workspace"
    if not os.path.isfile(full):
        return f"error: no such file: {path}"
    with open(full, encoding="utf-8", errors="replace") as f:
        content = f.read(MAX_CHARS + 1)
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + "\n... (truncated)"
    return content
