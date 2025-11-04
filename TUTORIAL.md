```markdown
# Hermes AI Logistics Assistant

Hermes AI is a PandasAI-powered logistics assistant that lets you query shipment data in natural language, generate visualizations, run predictions and get recommendations.

This repository uses the "src/" layout with the package namespace `hermes` (source files live under `src/hermes`).

## Quick start (development / local)

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

2. Install editable (development) install:

```bash
pip install -e .
```

An editable install allows you to edit code in `src/hermes` and immediately use the package without reinstalling.

3. Run the app:

- Option A — using the console script (recommended after install):

```bash
hermes-ai
```

- Option B — using python -m if you prefer not to rely on entry points:

```bash
python -m hermes.app
```

Note: The console script `hermes-ai` is defined in `pyproject.toml` and expects that `src/hermes/app.py` exposes a `main()` function that launches the Gradio UI. If you haven't defined `main()` yet, run with `python -m hermes.app` until you add it.

...
```