# Hermes — Copilot / AI Agent Instructions

Target: quick, actionable guidance for AI agents contributing to Hermes (an LLM-driven logistics analytics prototype).

## Repository Snapshot

**Code package:** `src/hermes/`
- **Key modules:** 
  - `app.py` — HermesApp orchestrator (data loading, query dispatch)
  - `ui_agent.py` — Gradio 5.48.0 chat interface with agent reasoning display
  - `router.py` — LLM-based query intent classification (prediction/visualization/stats/etc.)
  - `analytics.py` — Statistics, predictions, ML models (sklearn-based)
  - `visualizer.py` — Chart generation with cache-busting, PIL image handling
  - `autoviz.py` — Auto-visualization heuristics (response→chart spec→render)
  - `models.py` — Pydantic type-safe response schemas
  - `config.py` — PandasAI/LiteLLM config, paths, logger setup
  - `prompts.py` — LLM prompt templates (classification, visualization, stats, prediction)
  - `semantic.py` — Semantic dataset registration for LLM reasoning

**Data:** `data/` (CSVs: `shipments.csv`, `shipments_1000.csv`, `shipment_questions_500.csv`)
**Artifacts:** `exports/charts/` (PNG outputs), `logs/`, `.cache/`
**Entrypoint:** Console script `hermes` → `hermes.app:main` (pyproject.toml)

## Big Picture and Flow

### Query Processing Pipeline
```
User Query (Gradio Chat) → HermesApp.handle_query()
                          ↓
                  QueryRouter.classify_query()
                  - Try: LLM classification (JSON extraction)
                  - Fallback: regex pattern matching
                          ↓
                  Intent detected (visualization/prediction/stats/general/comparison)
                          ↓
              Dispatch to handler: _handle_{intent}_chat()
                          ↓
        ┌───────────────────────┬──────────────────┬──────────────────┐
        ↓                       ↓                  ↓                  ↓
   SmartDataframe.chat()  HermesAnalytics    HermesVisualizer    auto_visualize()
   (PandasAI LLM)        (ML + stats)       (PNG generation)    (heuristic charts)
        ↓                       ↓                  ↓                  ↓
   LLM Response          predictions/stats   chart_path           chart_path
        ↓                       ↓                  ↓                  ↓
        └───────────────────────┴──────────────────┴──────────────────┘
                                ↓
                        Format Response (Pydantic models)
                                ↓
                  Return to Gradio UI with reasoning logs
```

### Key Design Patterns

1. **LLM-Centric:** PandasAI SmartDataframe with LiteLLM (local/remote LLM) is the engine. Cache **disabled** to force fresh analysis per query.

2. **Type Safety:** All handler responses use Pydantic models (`BaseResponse` subclasses: `VisualizationResponse`, `PredictionResponse`, `StatisticsResponse`, etc.). Validates before returning to UI.

3. **Agent Reasoning:** `LLMReasoningCapture` (in `ui_agent.py`) intercepts logs to show model thinking in sidebar. Filters keywords: "query", "analysis", "response", "detected", "classified", "thinking", "result".

4. **Cache Busting:** `visualizer.py` ensures each visualization query generates a fresh chart. Old charts auto-cleaned (keeps last 5). PandasAI config: `enable_cache=False, save_charts=True`.

5. **Fallback Classification:** Router first tries LLM classification. If LLM fails or returns invalid JSON, falls back to regex patterns in `intent_patterns` dict.

6. **Auto-Visualization:** `autoviz.py` applies heuristics (time-series → line, categorical+numeric → bar, etc.) when response data is tabular but visualization unclear.

## LLM / Runtime Configuration (Must-Read)

**Location:** `src/hermes/config.py`

### Critical Settings
- **LLM Endpoint:** `http://localhost:8001/v1` (default). Requires a running LLM server (Ollama, vLLM, etc.)
- **Cache Disabled:** `enable_cache=False` in `pai.config` — **DO NOT CHANGE**. Forces fresh LLM calls per query.
- **Chart Saving:** `save_charts=True, save_charts_path=CHARTS_DIR` — charts auto-saved to `exports/charts/`
- **Verbose Logging:** `verbose=True` — captures reasoning via `pandasai_logger` for `LLMReasoningCapture`

### SmartDataframe Creation
```python
from pandasai import SmartDataframe
from hermes.config import llm, SMART_DF_CONFIG

smart_df = SmartDataframe(df, config=SMART_DF_CONFIG)
# Result: LLM-enabled dataframe with cache disabled
```

### Testing & Mocking
- **Unit tests:** Mock `SmartDataframe.chat()` to return deterministic strings/JSON. Avoids network calls.
- **Integration tests:** Point LLM endpoint to test server or mock endpoint (e.g., `http://localhost:8001/v1`)
- **CI environment:** Use mocks; do not rely on external LLM in CI pipeline.

### Debugging LLM Issues
- If `smart_df.chat()` fails: Check endpoint reachability with `curl http://localhost:8001/v1/models`
- If response is invalid JSON: Router logs raw output. Check `logs/` for reasoning traces.
- If classification confidence is low: LLM may be uncertain. Router falls back to regex patterns.

## Prompt & Parsing Conventions (Important)

### Classification Prompt
- **File:** `prompts.py`, key `'classification_intent'`
- **Expected Output:** JSON only: `{"intent":"visualization","confidence":0.92}`
- **Router Behavior:** Defensive JSON extraction — if LLM returns extra text, router searches for `{...}` substring
- **Allowed Intents:** `prediction`, `recommendation`, `visualization`, `comparison`, `statistics`, `general`

### Special Tokens (Preserve in Prompts)
- `<TIME_CONTEXT>...</TIME_CONTEXT>` — wraps temporal metadata (e.g., max date in dataset)
- `/no_think` — marker to disable reasoning tokens in some LLM configs

### Handler Return Contract
All `_handle_*_chat()` methods must return tuple:
```python
(formatted_html: str, chart_path: Optional[str], stats_dict: dict, preview_df: pd.DataFrame, chat_history: list)
```
- **formatted_html:** Markdown-formatted response text
- **chart_path:** File path to PNG or None
- **stats_dict:** Dictionary of key metrics (for display/export)
- **preview_df:** DataFrame sample (first 5 rows or aggregated summary)
- **chat_history:** Updated message history

### Response Type Safety
All responses validated via Pydantic models in `models.py`:
```python
from hermes.models import VisualizationResponse, PredictionResponse

response = VisualizationResponse(
    chart=ChartData(path="/path/to/chart.png", caption="..."),
    reasoning="..."
)
# Pydantic ensures all fields are valid before returning to UI
```

## Where to Change Behavior

### Adding New Intents
1. **Router:** Update `QueryRouter.intent_patterns` (regex fallback) in `router.py`
2. **Prompts:** Add template to `PROMPT_TEMPLATES` in `prompts.py` (for LLM classification)
3. **App Handler:** Implement `_handle_<intent>_chat()` in `app.py`
4. **Response Model:** Add new `BaseResponse` subclass in `models.py` (e.g., `CustomResponse`)
5. **UI:** Ensure Gradio components in `ui_agent.py` can display the new response type

### Modifying LLM Configuration
- **Primary:** Edit `llm = LiteLLM(...)` in `config.py`
  - Change `model`, `base_url`, `api_key`, `temperature`, `max_tokens`
  - Also update `pai.config.set({"llm": llm, ...})`
- **Alternative:** Set environment variables (`LLM_ENDPOINT`, `LLM_MODEL`)

### Changing Chart Generation
- **PandasAI Charts:** Config in `config.py`: `enable_cache`, `save_charts_path`
- **Heuristic Charts:** `autoviz.py` applies type detection logic in `_choose_chart_spec_from_df()`
- **Visualizer:** `visualizer.py` manages cache; `clear_cache_for_query()` before generating new charts

### Updating Intent Classifications
- **High confidence:** Add specific regex patterns to `intent_patterns` in `router.py`
- **Edge case:** Test fallback with `QueryRouter.classify_query(query_str)` and inspect `method` field

## Quick Debugging Checklist

- **Classification fails:** Log raw `smart_df.chat(...)` output. Router attempts to parse JSON substring. Inspect `QueryRouter.classify_query()` for LLM response.
- **Charts aren't produced:** Check `exports/charts/` directory, `CHARTS_DIR` path in `config.py`, and `pai.config['save_charts_path']`.
- **Reasoning not displayed:** Verify `LLMReasoningCapture` handler is attached to logger in `ui_agent.py`. Check log level filters (keywords: "query", "analysis", "response", "detected", "classified", "thinking", "result").
- **Prediction issues:** `HermesAnalytics.train_prediction_model` uses sklearn `LinearRegression` and `LabelEncoder` — ensure `route` and `warehouse` columns exist and have no unseen categories.
- **Cache reused:** If same chart appears across queries, check `enable_cache=False` in `config.py`. PandasAI config must NOT cache.
- **LLM endpoint unreachable:** Test with `curl http://localhost:8001/v1/models`. Ensure LLM server (Ollama, vLLM, etc.) is running.
- **Data parsing errors:** `HermesApp.load_data()` handles date coercion and `on_time` population. Check for unexpected column names or formats.

## Tests and Mocking Guidance

- **To unit-test routing/handlers:** Mock `SmartDataframe.chat` to return deterministic strings or JSON.
- **Example targets to test:**
  - `QueryRouter.classify_query()` JSON extraction (LLM returns text + JSON)
  - `HermesApp.load_data()` ensures date parsing and `on_time` population
  - `HermesAnalytics.train_prediction_model()` happy path and failure when required columns missing
  - `HermesVisualizer.get_latest_chart()` and cache-busting behavior
  - `LLMReasoningCapture.get_reasoning_markdown()` filters and formats logs correctly
  - `autoviz.auto_visualize()` heuristic chart spec selection for different data shapes

- **Integration test:** `test_integration.py` validates config (cache disabled), visualizer methods, and app initialization. Run with `pytest test_integration.py`.

- **Mock pattern:**
  ```python
  from unittest.mock import patch, MagicMock
  
  @patch('pandasai.SmartDataframe.chat')
  def test_handler(mock_chat):
      mock_chat.return_value = '{"intent":"visualization","confidence":0.92}'
      result = router.classify_query("show me a chart")
      assert result['intent'] == 'visualization'
  ```

## Code Examples (Copyable Patterns)

### Create a SmartDataframe
```python
from pandasai import SmartDataframe
from hermes.config import llm, SMART_DF_CONFIG

smart_df = SmartDataframe(df, config=SMART_DF_CONFIG)
# Result: LLM-enabled dataframe with cache disabled
```

### Use Classification Prompt
```python
from hermes.prompts import PROMPT_TEMPLATES
from hermes.router import QueryRouter

router = QueryRouter()
result = router.classify_query("show me delays by warehouse", smart_df=smart_df)
# Returns: {'intent': 'visualization', 'confidence': 0.92, 'method': 'llm'}
```

### Build a Response with Pydantic Validation
```python
from hermes.models import VisualizationResponse, ChartData

response = VisualizationResponse(
    chart=ChartData(path="/exports/charts/chart.png", caption="Delays by warehouse"),
    reasoning="Classified as visualization intent (0.92 confidence)"
)
# Pydantic validates; raises if path doesn't exist or required fields missing
```

### Add Auto-Visualization Fallback
```python
from hermes.autoviz import auto_visualize

# Try PandasAI chart, then fall back to heuristic
chart_result = auto_visualize(response_df, df_fallback=df, prompt_hint="Top routes by delay")
if chart_result:
    chart_path = chart_result['chart_path']
    caption = chart_result['caption']
```

### Capture LLM Reasoning
```python
from hermes.ui_agent import LLMReasoningCapture
import logging

reasoning_capture = LLMReasoningCapture()
logging.getLogger().addHandler(reasoning_capture)

# ... run query ...

markdown_display = reasoning_capture.get_reasoning_markdown()
reasoning_capture.clear()  # Reset for next query
```

## Common Pitfalls & Gotchas

1. **LLM endpoint URL and API key** in `config.py` may be placeholders. Running without a reachable LLM causes network errors; prefer mocking during CI.
2. **Chart-saving** relies on PandasAI's chart hooks; if `visualizer.get_latest_chart()` returns None, check `exports/charts/` and logs.
3. **Date parsing:** `load_data` coerces `date` and fills `on_time` for missing columns — altering this behavior affects analytics and prediction.
4. **Pydantic validation:** `ChartData.path` validator checks file existence. If you build responses with paths that don't exist yet, create files first or test files offline.
5. **Cache must be disabled:** If `enable_cache=True` in `config.py`, old charts reuse and reasoning becomes stale. Must be `enable_cache=False`.
6. **Intent JSON extraction:** If LLM returns malformed JSON, router logs a warning and falls back to regex. Check `logs/` if classification seems wrong.
7. **Auto-viz heuristics:** `autoviz.py` assumes numerical columns are measures and objects/datetime are dimensions. May generate incorrect charts for unusual data shapes.

## When Changing Prompts or Handler Contracts

- Update `prompts.py` and add tests that assert the expected LLM output shapes (JSON for classification; caption/text for visualization/statistics).
- Keep the handler return contract intact: `(html, chart_path|None, stats, preview_df, chat_history)`.
- If modifying `BaseResponse` subclasses in `models.py`, update UI components in `ui_agent.py` to render new fields correctly.
- Test LLM-dependent changes with mocks first to ensure robustness before running against live LLM endpoints.

If anything here is unclear or you'd like: (A) I can extend this to an onboarding README with setup commands and exact pip requirements; (B) add unit tests that mock `SmartDataframe.chat` for router and app handlers; or (C) create a small CI workflow that runs a smoke test (mocked) on push.

## Dev Mode & Rapid Iteration

### Auto-Reload Watch Script
- Located at repo root: `dev.py` uses `watchgod` for file watching
- Installation: `pip install watchgod` (or add to dev dependencies)
- Run: `python dev.py` (auto-restarts on source changes)
- Observes repository root; re-runs `hermes.app:main` on any `.py` save
- Startup errors displayed in console; fix and save to auto-restart

### Running the App
```bash
# Production: console script (recommended)
hermes

# Dev with auto-reload
python dev.py

# Manual restart for debugging
python -m hermes.app
```

### Key Dev Tools
- **Logging:** Check `logs/hermes.log` for detailed reasoning traces
- **Charts:** View cached/generated PNGs in `exports/charts/`
- **PandasAI Verbose:** `config.py` sets `verbose=True` for detailed LLM reasoning
- **Integration tests:** `pytest test_integration.py` validates core functionality

### UI Notes
- **Gradio 5.48.0:** Chat uses `type='messages'` format (not deprecated `type='messages'` with tuples)
- **Chart display:** Charts returned as file paths (not PIL objects) to avoid browser caching
- **Reasoning sidebar:** Powered by `LLMReasoningCapture` handler capturing specific keywords
- **Ensure Pillow installed:** Required for chart rendering in Gradio
