# Resonate Project Guidelines & Rules

## Testing Standards

- **Test Execution Command**: Always use `uv run pytest` to execute backend tests.
- **Core Principles**:
  - **No Third-Party Testing**: Do NOT test vendor or third-party code — test only our own application code.
  - **No Trivial Code Testing**: Do NOT test trivial or obvious code (e.g., `2+2=4`).
  - **Cover Happy & Error Paths**: Test both the happy path and the broken/error path.
  - **No Network Calls in Tests**: Do NOT make network calls in tests; mock all external APIs (Last.fm, MusicBrainz, Discogs, Plex).
  - **Write Rigorous Assertions**: Write meaningful tests — if bad code could easily pass the test, design a more rigorous assertion.
  - **App Logic Focus**: Ignore guarding against external infrastructure failures (hardware failure, network latency, etc.); focus strictly on our own application logic.
  - **Failure Handling**: If a test fails, do NOT delete it. Add smaller, focused unit tests to isolate the root cause.

## Git & Workflow Guidelines

- **Feature Branches**: Always work in a feature branch (e.g. `feat/...`, `fix/...`, `refactor/...`), never work or commit directly on `main`.
- **No Gemini / Agent Files in Repo**: Never add, stage, or commit any Gemini-related or agent-specific files (e.g. `.agent/`, `.gemini/`, prompt templates, debugging scratch files, or assistant metadata) into the repository.
- **Manual Verification Steps**: Always provide explicit, copy-pasteable manual verification steps (CLI commands, expected outputs, or inspection checks) alongside automated test results for all major changes.



