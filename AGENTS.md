# Agent Guide for Episodic Memory Pipeline

This document orients agentic coding assistants to the repo.
Scope: repository root and all subdirectories.

## Build, Lint, Test Commands

### Setup / Build
- `make install` installs the package in editable mode.
- `make install-dev` installs dev extras (pytest, ruff, mypy).
- `pip install -e .` is equivalent to `make install`.
- `pip install -e ".[dev]"` is equivalent to `make install-dev`.

### Lint / Format
- `make lint` runs `ruff check` on `src/`, `tests/`, `cli.py`, `config.py`.
- `make format` runs `ruff format` and `ruff check --fix`.
- `ruff check src/ tests/ cli.py config.py` and `ruff format src/ tests/ cli.py config.py` are direct equivalents.

### Tests
- `make test` runs fast tests only.
- `make test-slow` includes slow tests (model downloads).
- `make test-cov` runs coverage with HTML and terminal output.
- `pytest tests/ -v` is the canonical fast test command.

### Running a Single Test
- `pytest tests/test_file.py::test_name` runs a single test.
- `pytest tests/test_file.py -v` runs only a single file.
- `pytest -k "keyword"` runs tests matching the keyword.
- Slow tests are marked `@pytest.mark.slow` and require `--run-slow`.

## Codebase Structure
- `src/bootstrap.py` wires providers and shared components.
- `src/cli/` Click app/commands; `src/cli/render.py` Rich helpers.
- `src/services/` service layer (diagnostics/ingestion/retrieval/evaluation), no UI.
- `src/models/` Pydantic Episode/Fact/Summary with serialization helpers.
- `src/storage/` SQLite + FAISS (`database.py`, `vector_store.py`).
- `src/ingestion/`, `src/consolidation/`, `src/retrieval/` core pipelines.
- `src/llm/`, `src/embeddings/` provider interfaces; `src/prompts/` templates.
- `src/utils/llm_sanitize.py` sanitization and safety helpers.
- `tests/` pytest suites; `scripts/` bootstrap utilities.
- `schema.sql` DB schema; `config.py` runtime config; `cli.py` legacy entrypoint.
- `demo_data/` synthetic-only; `data/` and `runs/` are generated.

## Code Style Guidelines

### Formatting and Linting
- Formatting is handled by `ruff format`.
- Linting is handled by `ruff check`.
- Line length is 100 and target Python is 3.10.
- Do not introduce new formatters or linters.

### Imports
- Use three import groups: standard library, third-party, local.
- Ruff/isort enforces ordering and spacing.
- Prefer explicit imports; use `TYPE_CHECKING` for type-only imports.
- Within `src/`, relative imports are common (e.g., `from ..models`).
- CLI commands can use absolute `from src...` imports.

### Types
- Add type hints for public functions and methods.
- Use `Optional[...]` (or `| None`) when a value may be missing.
- Prefer concrete collections (`list[str]`, `dict[str, Any]`).
- Avoid `Any` unless required by external APIs.
- Use docstrings for modules/classes/public methods with `Args:` and `Returns:`.
- Pydantic models live in `src/models/` and should stay there.

### Naming
- Classes use `PascalCase`.
- Functions/variables use `snake_case`.
- Constants use `UPPER_SNAKE_CASE`.
- Use descriptive names: avoid single-letter variables.
- Identifiers end with `_id` where applicable (e.g., `episode_id`).

### Error Handling
- Use exceptions for exceptional cases; avoid silent failures.
- Log errors with `logging.getLogger(__name__)` before re-raising; avoid broad `except Exception`.
- Prefer context managers for resource cleanup (`with` for DB connections) and validate inputs early.
- Return structured errors in services.

### CLI vs Service Boundaries
- CLI code (`src/cli/`) may import Click/Rich and render output.
- Service modules (`src/services/`) should stay UI-free.
- Services return dataclasses or plain dicts, not Rich/Click objects.
- Keep CLI argument parsing separate from service logic.

### Data Models
- Use Pydantic `BaseModel` for `Episode`, `Fact`, `Summary`.
- Keep serialization logic inside models (`to_db_row`, `from_db_row`).
- Prefer dataclasses for simple result containers in services.
- Avoid circular imports by using forward references and `TYPE_CHECKING`.

### Database & Storage
- SQLite access is centralized in `src/storage/database.py`.
- Use parameterized SQL; do not build SQL with f-strings.
- Keep DB mutations inside the `Database` class.
- Ensure topic counters stay consistent when updating episodes.

### LLM / Embedding Integration
- LLM providers and embedding providers are in `src/llm/` and `src/embeddings/`.
- Guard against model/network failures; bubble up actionable errors.
- Keep prompt templates in `src/prompts/`.
- Sanitization helpers live in `src/utils/llm_sanitize.py`.

### Testing Guidelines
- Tests live in `tests/` and follow `test_*.py` naming.
- Use pytest fixtures where shared setup is needed.
- Mark slow tests with `@pytest.mark.slow`.
- Keep tests deterministic; avoid requiring external services.

### Configuration
- Runtime settings come from `.env` or environment variables.
- Example values live in `env.example`.
- Do not commit real API keys or user data.

## Operational Hygiene
- Use module-level loggers and `%s` placeholders; avoid `print()` in services.
- Keep logs at info/debug unless reporting errors; surface actionable user errors.
- Avoid touching `data/`/`runs/`; keep `demo_data/` synthetic only.
- Do not add large binaries; `schema.sql` is the schema source of truth.
- Keep changes minimal; avoid unrelated refactors or formatting churn.
- Do not commit secrets/local artifacts; prefer incremental changes.

## Production Best Practices

### Reliability and Resilience
- Fail fast on invalid inputs; return structured errors in services.
- Use idempotent writes or explicit checks for reprocessing.
- Prefer timeouts and clear error context on external calls.

### Security and Data Safety
- Treat all user input as untrusted; sanitize before persistence.
- Never log secrets, API keys, or raw PII.
- Keep secrets in `.env` or environment variables.

### Performance and Scalability
- Avoid N+1 queries; batch reads and bound top-k vector ops.
- Reuse clients/providers instead of re-instantiating per call.
- Prefer streaming or chunked processing for large inputs.

### Observability and Operability
- Emit logs with consistent message structures and context.
- Log error boundaries with enough context for remediation.
- Surface user-facing errors with actionable instructions.

### Testing and QA Discipline
- Add tests for non-trivial logic and edge cases.
- Prefer deterministic tests; mock LLM/embedding providers.
- Keep slow tests isolated and opt-in (`--run-slow`).

## Cursor / Copilot Rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` were found.

