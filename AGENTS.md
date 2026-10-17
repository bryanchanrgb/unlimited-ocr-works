# Agent Instructions

- Make the smallest code changes that solve the task; avoid unnecessary additions.
- Before implementing a feature, inspect the existing codebase and reuse or extend existing functionality rather than duplicating it.
- Create only files and directories needed for the requested change. Do not add placeholders or speculative project structure.
- Use `pathlib` for filesystem paths, and write code that works on Windows, macOS, and Linux. Avoid platform-specific path assumptions.
- Keep code compatible with the Python version declared in `pyproject.toml` (currently Python 3.14 or newer).
- After code changes, run relevant tests when a test suite is available. Use the project's configured linter and formatter; prefer Ruff for Python linting and formatting when no tools are configured. Do not add tooling solely to validate a change unless requested.
- Update the README or relevant documentation when a change affects user-facing behavior.
- Preserve raw input data; do not overwrite files under `data/raw`. Save derived or processed outputs under `data/processed`.
- `src` is for deployable code only. Use `scripts` or other folders for non-deployable code.
- A docstring is required for every function. A simple one-line description is sufficient, for more complex functions the docstring should explain the logic.
- Use `.env` for storing credentials. Never write credentials to other files.