"""Tool: save a note as a markdown file in the workspace notes/ folder."""

import os

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES_DIR = os.path.join(WORKSPACE, "notes")

TOOL = {
    "type": "function",
    "function": {
        "name": "write_note",
        "description": "Save a note as a markdown file in the workspace notes/ folder. Use when the user dictates something to remember, a todo, or asks to write something down.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Short kebab-case filename without extension, e.g. 'shopping-list'.",
                },
                "content": {
                    "type": "string",
                    "description": "The note content in markdown.",
                },
            },
            "required": ["filename", "content"],
        },
    },
}


def run(filename, content):
    safe = os.path.basename(filename).replace(" ", "-")
    if not safe:
        return "error: empty filename"
    if not safe.endswith(".md"):
        safe += ".md"
    os.makedirs(NOTES_DIR, exist_ok=True)
    path = os.path.join(NOTES_DIR, safe)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")
    return f"note saved to notes/{safe}"
