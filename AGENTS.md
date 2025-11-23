# Repository Guidelines

## Project Structure & Module Organization
- App entrypoint: `src/enemane_ai/app.py` (Streamlit UI). Core graph/PDF handling and Gemini client logic: `src/enemane_ai/analyzer.py`. Package initializer lives in `src/enemane_ai/__init__.py`.
- Tests reside in `tests/` (Pytest discovers via `test_*.py`). Temporary files are created under `tmp` when running tests; nothing should be written to the repo root.
- Use `src/` layout for imports (`enemane_ai.*`). Keep new utilities in the package to avoid path hacks.

## Build, Test, and Development Commands
- Sync deps (Python 3.13 assumed): `uv sync --dev`.
- Run the app locally: `uv run streamlit run src/enemane_ai/app.py`.
- Test suite: `uv run pytest` (quiet mode via `-q` set in `pyproject.toml`).
- Lint/format: `uv run ruff check` and `uv run ruff format`.
- Type check: `uv run mypy src`.
- Install git hooks: `uv run pre-commit install`.

## Coding Style & Naming Conventions
- Python only; use 4-space indentation and snake_case for modules, functions, and variables. Classes use CapWords.
- Ruff enforces formatting (line length 100, double quotes, spaces for indent). Avoid trailing commas where they break Ruff’s magic-comma rule.
- Keep Streamlit UI text bilingual-friendly (current UI uses Japanese copy). Handle optional config gracefully—prefer explicit error messages (see `resolve_gemini_client`).

## Testing Guidelines
- Framework: Pytest (`tests/test_*.py`). Mirror behavior with sandboxed fakes (see `DummyLLM` pattern) to avoid real API calls.
- For new features, add focused unit tests alongside existing ones. Prefer in-memory images/PDFs to keep tests fast and deterministic.
- Aim for coverage of error paths (missing `GEMINI_API_KEY`, PDF parsing failures) and LLM call contracts.
- Follow t-wada style TDD: write a failing test first, make the minimal change to pass, then refactor with tests staying green.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative summaries (e.g., “Add Gemini-based graph analyzer with typing and tooling”). Group related changes per commit.
- PRs: include a brief description, test results (`uv run pytest`, lint/format if relevant), and screenshots/GIFs for UI changes. Link issues when applicable.
- GitHub CLI (`gh`) is available; prefer `gh pr create` / `gh pr view` / `gh pr checkout` for PR workflows after `gh auth status` confirms login.

## Security & Configuration Tips
- Required secret: `GEMINI_API_KEY` (env var or `.streamlit/secrets.toml`). Never commit secrets or `.streamlit/secrets.toml`.
- PDFs are rendered locally with `pypdfium2`; avoid untrusted files when debugging. Keep temporary files within `TemporaryDirectory` contexts as in `app.py`.
- Pin Python to 3.13 to match tooling and type targets.***
