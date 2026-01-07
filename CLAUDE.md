# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

enemane-ai is a Streamlit app for analyzing energy-related graphs, PDFs, and CSV data using Google Gemini AI. It generates AI comments for power usage charts, calendar analysis, and article summarization for energy management reports.

## Commands

```bash
# Install dependencies (Python 3.13 required)
uv sync --dev

# Run the app
uv run streamlit run src/enemane_ai/app.py

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_analyzer.py::test_function_name -v

# Lint and format
uv run ruff check
uv run ruff format

# Type check
uv run mypy src

# Install git hooks
uv run pre-commit install
```

## Architecture

### Source Layout (`src/enemane_ai/`)

- **app.py**: Streamlit UI with three main tabs:
  - Graph Analysis: Analyzes power usage graphs with supplementary CSV data
  - Power Calendar Analysis: Analyzes 30-min interval power data PDFs
  - Article Search: Collects and summarizes energy-related articles via DuckDuckGo

- **analyzer.py**: Core logic including:
  - `GeminiGraphLanguageModel`: Wrapper for Google Gemini API (`gemini-3-pro-preview` default, `gemini-2.5-flash` for fast operations)
  - CSV parsers: `parse_monthly_report_csv`, `parse_power_30min_csv`, `parse_temperature_csv_for_comparison`
  - PDF handling: `pdf_to_images` (uses pypdfium2)
  - Article collection: `collect_relevant_articles`, `judge_article_relevance`, `summarize_article`
  - Prompt templates: `PRESET_PROMPT`, `CALENDAR_ANALYSIS_PROMPT`, `OUTPUT_FORMAT_INSTRUCTION`

### Data Flow

1. User uploads files (images, PDFs, CSVs) via Streamlit
2. Files are saved to `TemporaryDirectory`
3. CSVs parsed into structured dataclasses (`MonthlyReportData`, `MonthlyPowerCalendarData`, `MonthlyTemperatureSummary`)
4. Context strings built from parsed data
5. Gemini API called with image/text + prompt
6. JSON responses parsed and displayed as tables
7. Results exportable as CSV (BOM-prefixed UTF-8 or CP932)

### Key Dataclasses

- `MonthlyReportData`: Previous year monthly report with circuit-level power data
- `MonthlyPowerCalendarData`: 30-min interval power data aggregated by day
- `ArticleFetchResult` / `ArticleCollectionResult`: Web article scraping results

## Development Guidelines

### TDD Approach
Follow t-wada style TDD: write failing test first, minimal implementation to pass, then refactor.

### Testing
- Tests in `tests/` directory, use Pytest
- Use `DummyLLM` pattern to mock Gemini API calls
- Create in-memory images/PDFs for fast, deterministic tests

### Code Style
- Ruff enforces formatting (line length 100, double quotes)
- Snake_case for functions/variables, CapWords for classes
- UI text is Japanese

### Required Secret
`GEMINI_API_KEY` must be set in `.streamlit/secrets.toml` or as environment variable.

### PR Guidelines
Write PR descriptions in Japanese. Include test results and screenshots for UI changes.
