from __future__ import annotations
import os
import hashlib
import time
import logging
from pathlib import Path
from typing import Optional, Any, Dict
import pandas as pd

import plotly.express as px

from .config import CHARTS_DIR
from .prompts import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s)[:60].strip("-")

def _make_chart_filename(prefix: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{_slugify(prefix)}_{ts}.png"
    return str(Path(CHARTS_DIR) / name)

def _df_from_response(response: Any, df_fallback: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Try to extract a pandas DataFrame from different response shapes:
    - if response is already a DataFrame -> return
    - if response is dict with 'data' or 'table' -> convert to DataFrame
    - if response is list of dicts -> DataFrame
    - else return None
    """
    try:
        if isinstance(response, pd.DataFrame):
            return response
        if isinstance(response, dict):
            # common keys
            for key in ("table", "data", "rows"):
                if key in response and isinstance(response[key], (list, dict)):
                    return pd.DataFrame(response[key])
            # if dict of columns
            if all(isinstance(v, (list, tuple)) for v in response.values()):
                return pd.DataFrame(response)
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict):
                return pd.DataFrame(response)
            # if list of lists and fallback provided, try to use columns from fallback
        # fallback: if df_fallback provided and response is a small summary (e.g., top categories), return None; we'll use heuristics on df_fallback
    except Exception as e:
        logger.debug("Can't convert response to df: %s", e)
    return None

def _choose_chart_spec_from_df(df: pd.DataFrame) -> Dict[str, str]:
    """
    Heuristic for chart spec:
    Returns dict {type: 'line'|'bar'|'hist'|'scatter', x: col, y: col, color: optional}
    """
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    datetime_cols = df.select_dtypes(include='datetime').columns.tolist()

    spec = {"type": "table", "x": None, "y": None, "color": None}
    # time series
    if datetime_cols and num_cols:
        spec.update({"type": "line", "x": datetime_cols[0], "y": num_cols[0]})
        return spec
    # categorical counts or aggregations
    if cat_cols and num_cols:
        spec.update({"type": "bar", "x": cat_cols[0], "y": num_cols[0]})
        return spec
    if cat_cols:
        # count of category
        spec.update({"type": "bar", "x": cat_cols[0], "y": None})
        return spec
    if len(num_cols) >= 2:
        spec.update({"type": "scatter", "x": num_cols[0], "y": num_cols[1]})
        return spec
    if num_cols:
        spec.update({"type": "hist", "x": num_cols[0], "y": None})
        return spec
    return spec

def render_chart_from_spec(df: pd.DataFrame, spec: Dict[str, str], title: str = "") -> Optional[str]:
    """
    Render a plotly chart from spec and save as PNG. Returns chart path or None.
    """
    try:
        os.makedirs(CHARTS_DIR, exist_ok=True)
        filename = _make_chart_filename(title or spec.get("type", "chart"))
        chart_type = spec.get("type")
        x = spec.get("x")
        y = spec.get("y")
        color = spec.get("color", None)

        fig = None
        if chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color, title=title)
        elif chart_type == "bar":
            if y:
                fig = px.bar(df, x=x, y=y, color=color, title=title)
            else:
                # counts
                ct = df[x].value_counts().reset_index()
                ct.columns = [x, "count"]
                fig = px.bar(ct, x=x, y="count", title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, title=title)
        elif chart_type == "hist":
            fig = px.histogram(df, x=x, title=title)
        else:
            # unsupported -> no chart
            return None

        # save as png (requires kaleido)
        fig.write_image(filename, engine="kaleido")
        return filename
    except Exception as e:
        logger.exception("Failed to render chart: %s", e)
        return None

def auto_visualize(response: Any, df_fallback: Optional[pd.DataFrame] = None, prompt_hint: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Main entry. Try to find a DataFrame in response or fallback to df_fallback,
    propose a chart spec (heuristic) and render it. Returns dict {chart_path, caption, spec}
    or None if nothing visualizable.
    """
    # 1) if PandasAI already saved charts, caller should check CHARTS_DIR first (we do not here)
    df = _df_from_response(response, df_fallback=df_fallback)
    used_df = df or df_fallback
    if used_df is None or used_df.empty:
        return None

    # 2) try to let LLM propose a chart spec if prompt_hint provided (optional)
    # For simplicity and safety, we apply heuristic spec:
    spec = _choose_chart_spec_from_df(used_df)

    # 3) render
    title = (prompt_hint or "Auto Visualization").strip()
    chart_path = render_chart_from_spec(used_df, spec, title=title)
    if not chart_path:
        return None

    # 4) produce a small caption
    caption = f"Auto-generated {spec.get('type')} chart"
    if spec.get("x") and spec.get("y"):
        caption += f" of {spec['y']} vs {spec['x']}."
    elif spec.get("x"):
        caption += f" for {spec['x']}."
    return {"chart_path": chart_path, "caption": caption, "spec": spec}