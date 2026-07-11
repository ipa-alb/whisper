"""Tool: list files in the workspace."""

import os

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {".git", ".venv", "__pycache__"}

TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List the files and folders in the workspace, optionally inside a subdirectory.",
        "parameters": {
            "type": "object",
            "properties": {
                "subdir": {
                    "type": "string",
                    "description": "Subdirectory relative to the workspace root. Omit for the root.",
                }
            },
            "required": [],
        },
    },
}


def run(subdir="."):
    base = os.path.realpath(os.path.join(WORKSPACE, subdir))
    if not (base == WORKSPACE or base.startswith(WORKSPACE + os.sep)):
        return "error: path escapes the workspace"
    if not os.path.isdir(base):
        return f"error: no such directory: {subdir}"
    entries = []
    for name in sorted(os.listdir(base)):
        if name in SKIP:
            continue
        path = os.path.join(base, name)
        if os.path.isdir(path):
            entries.append(f"{name}/")
        else:
            entries.append(f"{name} ({os.path.getsize(path)} bytes)")
    return "\n".join(entries) if entries else "(empty)"
