# Hermes — Copilot / AI Agent Instructions

Target: quick, actionable guidance for AI agents contributing to Hermes (an LLM-driven logistics analytics prototype).

1) Repository snapshot
- Code package: `src/hermes` — key modules: `app.py` (HermesApp), `analytics.py`, `visualizer.py`, `router.py`, `prompts.py`, `config.py`, `semantic.py`, `ui.py`.
- Data: `data/` (sample CSVs: `shipments.csv`, `shipment_questions_500.csv`).
- Artifacts: `charts/` (saved PNGs), `logs/`, `.cache/`.
- Entrypoint: console script `hermes` -> `hermes.app:main` (defined in `pyproject.toml`).

2) Big picture and flow
- HermesApp (in `src/hermes/app.py`) is the orchestrator: loads CSVs -> constructs `pandas.DataFrame` and `pandasai.SmartDataframe` -> routes NL queries via `QueryRouter` -> dispatches to handlers that call `HermesAnalytics` or `HermesVisualizer` or use `SmartDataframe.chat` directly.
- LLM + PandasAI are central: `SmartDataframe.chat(...)` executes or reasons over the dataframe. Many handlers expect the LLM to return short captions, JSON intent, or produce charts saved to `charts/`.
- Semantic registration: `register_semantic_dataset` (in `semantic.py`) registers the loaded DataFrame for improved LLM reasoning; path is `hermes/shipments` by default.

3) LLM / runtime notes (must-read)
- Configuration in `src/hermes/config.py`:
  - Uses `pandasai_litellm.LiteLLM` pointed at `http://localhost:8001/v1` by default. You must run or provide a compatible LLM endpoint or change `llm` config.
  - `pai.config` is set to save charts (`save_charts=True`) and to use `CHARTS_DIR`.
- SmartDataframe is created with `config={"llm": llm, "enable_cache": True}`. Tests and local runs should mock `SmartDataframe.chat` to avoid network calls.

4) Prompt and parsing conventions (important)
- Classification: `PROMPT_TEMPLATES['classification_intent']` expects the LLM to return ONLY JSON like `{"intent":"visualization","confidence":0.92}`. However, the router is defensive: it will try to extract a JSON substring if the model returns extra text.
- Special tokens used by code: `<TIME_CONTEXT>...</TIME_CONTEXT>` and the string `/no_think` appear in prompts. Keep them intact when editing prompts.

5) Where to change behavior
- Add/update intents:
  - Update `QueryRouter.intent_patterns` (regex fallback) and `PROMPT_TEMPLATES['classification_intent']` to reflect new intent names.
  - Implement a `_handle_<intent>_request(self, prompt: str)` method in `HermesApp` that returns the same tuple shape: (formatted_html, chart_path_or_None, stats_dict, preview_df, chat_history).
- Modify LLM config: change `llm` in `src/hermes/config.py` and `pai.config.set(...)` settings.

6) Quick debugging checklist
- If classification fails: log raw `smart_df.chat(...)` output (router attempts to parse JSON substring). Inspect in `QueryRouter.classify_query`.
- If charts aren’t produced: check `charts/` directory, `CHARTS_DIR` path in `config.py`, and `pai.config['save_charts_path']`.
- For prediction issues: `HermesAnalytics.train_prediction_model` uses sklearn's `LinearRegression` and `LabelEncoder` — ensure `route` and `warehouse` columns exist and have no unseen categories at prediction time.

7) Tests and mocking guidance
- To unit-test routing/handlers, mock `SmartDataframe.chat` to return deterministic strings or JSON.
- Example targets to test:
  - `QueryRouter.classify_query` JSON extraction behavior (LLM returns text + JSON). 
  - `HermesApp.load_data` ensures date parsing and `on_time` population.
  - `HermesAnalytics.train_prediction_model` happy path and failure when required columns missing.

8) Small code examples (copyable patterns)
- Create a SmartDataframe (used throughout):

```py
from pandasai import SmartDataframe
smart_df = SmartDataframe(df, config={"llm": llm, "enable_cache": True})
```

- Classification prompt usage (router): uses `PROMPT_TEMPLATES['classification_intent']` and expects JSON.

9) Common pitfalls / gotchas
- LLM endpoint URL and API key in `config.py` may be placeholders. Running without a reachable LLM causes network errors; prefer mocking during CI.
- Chart-saving relies on PandasAI's chart hooks; if `visualizer.get_latest_chart()` returns None, check `charts/` and logs.
- Date parsing: `load_data` coerces `date` and fills `on_time` for missing columns — altering this behavior affects analytics and prediction.

10) When you change prompts or handler contracts
- Update `prompts.py` and add tests that assert the expected LLM output shapes (JSON for classification; caption/text for visualization/statistics).
- Keep the handler return contract intact: (html, chart_path|None, stats, preview_df, chat_history).

If anything here is unclear or you'd like: (A) I can extend this to an onboarding README with setup commands and exact pip requirements; (B) add unit tests that mock `SmartDataframe.chat` for router and app handlers; or (C) create a small CI workflow that runs a smoke test (mocked) on push.

11) Dev mode (auto-reload)
- Quick helper: a small watcher script `dev.py` is included at the repo root. It uses `watchgod` to auto-restart the app when files change.
- Install: `pip install watchgod` (or add `watchgod` to your dev dependencies).
- Run in dev mode:

```bash
python dev.py
```

This runs the same entrypoint as the console script (`python -m hermes.app`) and restarts on source changes. It's intended for local iterative development so you don't need to relaunch the server after each edit.

Notes:
- The watcher observes the repository root by default. If you prefer another watcher (e.g., `watchmedo` from `watchdog`), you can replace the approach in `dev.py`.
- If the app fails to start (e.g., missing LLM endpoint or dependencies), the watcher will show the startup error in the console; fix the issue and the watcher will restart the app on the next code save.

Note: the UI was recently changed to a conversation-style `gr.Chatbot` and chart images are returned as in-memory PIL Images to avoid browser caching of overwritten PNG files. Ensure `Pillow` is installed when running the app (`pip install pillow`).
