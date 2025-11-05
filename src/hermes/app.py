"""
Application module for the hermes package.

Defines HermesApp (the application object used by UI) and main()
(entrypoint used by the console script `hermes`).

This module ties together router, analytics, visualizer, and semantic helpers.
It uses the PandasAI SmartDataframe for LLM-driven queries and preserves
the same handlers used by the earlier single-file implementation.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pandasai import SmartDataframe

from .config import CHARTS_DIR, DATA_DIR, QUESTIONS_FILE, SHIPMENTS_FILE, llm
from .analytics import HermesAnalytics
from .visualizer import HermesVisualizer
from .router import QueryRouter
from .semantic import register_semantic_dataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class HermesApp:
    """Main application class providing data loading, query processing and helpers."""

    def __init__(self) -> None:
        self.chat_history: list[Dict[str, Any]] = []
        self.current_df: Optional[pd.DataFrame] = None
        self.smart_df: Optional[SmartDataframe] = None
        self.analytics: Optional[HermesAnalytics] = None
        self.visualizer = HermesVisualizer(charts_dir=CHARTS_DIR)
        self.router = QueryRouter()

    # -----------------------
    # Data loading / helpers
    # -----------------------
    def get_csv_files(self) -> list[str]:
        """Return list of CSV files in the data directory."""
        return sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])

    def get_questions(self) -> list[str]:
        """Load sample questions list (if present)."""
        try:
            if os.path.exists(QUESTIONS_FILE):
                return pd.read_csv(QUESTIONS_FILE)["question"].astype(str).tolist()
        except Exception as e:
            logger.warning("Failed to load questions file: %s", e)
        return []

    def load_data(self, data_source_type: str, uploaded_file: Optional[Any], selected_file: Optional[str]) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load data from selected_file or uploaded_file.

        Returns (df, message). On failure returns (None, error_message).
        """
        try:
            if data_source_type == "Upload New Data CSV":
                if uploaded_file is None:
                    return None, "❌ Please upload a CSV file"
                data_path = uploaded_file.name
            else:
                if not selected_file:
                    return None, "❌ Please select a data file"
                data_path = selected_file

            df = pd.read_csv(data_path)

            # Ensure required basic columns and types
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if "delay_minutes" in df.columns and "on_time" not in df.columns:
                df["on_time"] = df["delay_minutes"].fillna(0).astype(int).apply(lambda x: 1 if x == 0 else 0)

            self.current_df = df

            # Register semantic dataset for better LLM reasoning (best-effort)
            try:
                register_semantic_dataset(df)
            except Exception as e:
                logger.warning("Semantic registration failed: %s", e)

            # Create SmartDataframe with configured LLM
            self.smart_df = SmartDataframe(df, config={"llm": llm, "enable_cache": True})
            self.analytics = HermesAnalytics(df)

            logger.info("Loaded %d records from %s", len(df), data_path)
            return df, f"✅ Loaded {len(df)} shipment records"
        except Exception as e:
            logger.exception("Data loading error")
            return None, f"❌ Error loading data: {e}"

    # -----------------------
    # Query processing
    # -----------------------
    def process_query(self, data_source_type: str, uploaded_file: Optional[Any], selected_file: Optional[str],
                      selected_question: Optional[str], user_prompt: Optional[str]) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Optional[pd.DataFrame], list[Dict[str, Any]]]:
        """
        Main query processing function used by the UI.

        Returns tuple: (formatted_response_html, chart_path_or_None, stats_dict, data_preview_df, chat_history)
        """
        # Ensure data loaded
        if self.current_df is None or self.smart_df is None:
            df, msg = self.load_data(data_source_type, uploaded_file, selected_file)
            if df is None:
                return msg, None, None, None, self.chat_history

        # Build prompt
        prompt = selected_question.strip() if selected_question else ""
        if user_prompt:
            prompt = user_prompt.strip() if not prompt else f"{prompt}\n{user_prompt.strip()}"
        if not prompt:
            return "❌ Please enter a question or select one from the list", None, None, None, self.chat_history

        try:
            # LLM-driven routing (pass SmartDataframe for LLM-based classification)
            classification = self.router.classify_query(prompt, smart_df=self.smart_df)
            intent = classification.get("intent", "general")
            confidence = classification.get("confidence", 0.0)
            method = classification.get("method", "fallback")
            force_chart = self.router.should_force_chart(prompt, smart_df=self.smart_df)

            logger.info("Routing: intent=%s confidence=%.2f method=%s force_chart=%s", intent, confidence, method, force_chart)

            # Dispatch to handlers
            if intent == "prediction":
                formatted, chart, stats, preview, history = self._handle_prediction_request(prompt)
            elif intent == "recommendation":
                formatted, chart, stats, preview, history = self._handle_recommendation_request(prompt)
            elif intent in ("visualization", "comparison"):
                formatted, chart, stats, preview, history = self._handle_visualization_request(prompt, force_chart)
            elif intent == "statistics":
                formatted, chart, stats, preview, history = self._handle_stats_request(prompt)
            else:
                formatted, chart, stats, preview, history = self._handle_general_query(prompt, force_chart)

            return formatted, chart, stats, preview, history

        except Exception as e:
            logger.exception("Query processing error")
            err = f"<div style='color:red;'><strong>Error processing query:</strong> {e}</div>"
            return err, None, None, None, self.chat_history

    # -----------------------
    # Handlers
    # -----------------------
    def _handle_prediction_request(self, prompt: str):
        """Handle prediction intent (uses internal ML pipeline)."""
        logger.info("Handling prediction request")
        result = self.get_predictions()
        stats = self.analytics.get_summary_stats() if self.analytics else {}
        self._add_to_history(prompt, result, "prediction")
        return result, None, stats, (self.current_df.head(10) if self.current_df is not None else None), self.chat_history

    def _handle_recommendation_request(self, prompt: str):
        """Handle recommendation intent."""
        logger.info("Handling recommendation request")
        result = self.get_recommendations()
        stats = self.analytics.get_summary_stats() if self.analytics else {}
        self._add_to_history(prompt, result, "recommendation")
        return result, None, stats, (self.current_df.head(10) if self.current_df is not None else None), self.chat_history

    def _handle_visualization_request(self, prompt: str, force_chart: bool = True):
        """Handle visualization intent by asking SmartDataframe to generate a chart."""
        logger.info("Handling visualization request")
        max_date = self._get_time_context()
        viz_prompt = f"<TIME_CONTEXT>Current date: {max_date}</TIME_CONTEXT>\n{prompt}\nPlease create and save a relevant visualization as a PNG. /no_think"
        response = self.smart_df.chat(viz_prompt)
        chart_path = self.visualizer.get_latest_chart()
        chart_for_ui = self._get_chart_for_ui(chart_path)

        stats = self.analytics.get_summary_stats() if self.analytics else {}
        response_html = self._beautify_response(response, title="📊 Visualization", footer="")
        self._add_to_history(prompt, response, "visualization", chart_path)
        return response_html, (chart_for_ui if chart_for_ui else chart_path), stats, (self.current_df.head(10) if self.current_df is not None else None), self.chat_history

    def _handle_stats_request(self, prompt: str):
        """Handle stats intent by asking PandasAI and returning summary stats as well."""
        logger.info("Handling statistics request")
        max_date = self._get_time_context()
        full_prompt = f"<TIME_CONTEXT>Current date: {max_date}</TIME_CONTEXT>\n{prompt} /no_think"
        response = self.smart_df.chat(full_prompt)
        stats = self.analytics.get_summary_stats() if self.analytics else {}
        footer = f"Total shipments: {stats.get('total_shipments', 0)}, Delayed: {stats.get('delayed_shipments', 0)}" if stats else ""
        response_html = self._beautify_response(response, title="📊 Statistics", footer=footer)
        self._add_to_history(prompt, response, "statistics")
        return response_html, None, stats, (self.current_df.head(10) if self.current_df is not None else None), self.chat_history

    def _handle_general_query(self, prompt: str, force_chart: bool = False):
        """Default handler that forwards the prompt to PandasAI for execution."""
        logger.info("Handling general query")
        max_date = self._get_time_context()
        full_prompt = f"<TIME_CONTEXT>Current date: {max_date}</TIME_CONTEXT>\n{prompt} /no_think"
        if force_chart:
            full_prompt += "\nPlease include a relevant visualization and save the chart."
        response = self.smart_df.chat(full_prompt)
        stats = self.analytics.get_summary_stats() if self.analytics else {}
        response_html = self._beautify_response(response, title="", footer="")
        
        chart_path = self.visualizer.get_latest_chart()
        chart_for_ui = self._get_chart_for_ui(chart_path)

        self._add_to_history(prompt, response, "general", chart_path)
        return response_html, (chart_for_ui if chart_for_ui else chart_path), stats, (self.current_df.head(10) if self.current_df is not None else None), self.chat_history

    # -----------------------
    # Utilities & features
    # -----------------------
    def _get_time_context(self) -> str:
        """Extract max date from current dataframe for time context."""
        if self.current_df is not None and "date" in self.current_df.columns:
            try:
                return self.current_df["date"].max().strftime("%Y-%m-%d")
            except Exception:
                pass
        return ""
    
    def _get_chart_for_ui(self, chart_path: Optional[str]):
        """Convert chart path to PIL Image for UI display."""
        if not chart_path:
            return None
        try:
            from PIL import Image
            return Image.open(chart_path).copy()
        except Exception:
            return None
    
    def _add_to_history(self, prompt: str, response: str, intent: str, chart_path: Optional[str] = None):
        """Add entry to chat history."""
        entry = {
            "query": prompt,
            "response": str(response),
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        }
        if chart_path is not None:
            entry["chart_generated"] = True
            entry["chart_path"] = chart_path
        self.chat_history.append(entry)
    
    def _beautify_response(self, response_obj: Any, title: str = "", footer: str = "") -> str:
        """Return formatted response string for display in chat UI."""
        # Convert response to safe string and replace newlines with <br>
        response_str = str(response_obj).replace("\n", "<br>")
        
        # For chat-style interface, use simpler formatting
        if title:
            html = f"<strong>{title}</strong><br><br>{response_str}"
        else:
            html = response_str
            
        if footer:
            html += f"<br><br><small style='opacity:0.8;'>{footer}</small>"
        
        return html

    def get_predictions(self) -> str:
        """Train model and return formatted prediction results."""
        if self.analytics is None:
            return "❌ Please load data first"
        try:
            metrics = self.analytics.train_prediction_model()
            pred = self.analytics.predict_next_week()
            if not metrics or not pred:
                return "❌ Prediction failed"
            html = f"""<strong>🔮 Predictions for Next Week</strong><br><br>
Based on historical data analysis, here's what I forecast:<br><br>
📊 <strong>Model Performance:</strong><br>
• R² Score: {metrics['r2_score']:.3f}<br>
• RMSE: {metrics['rmse']:.2f} minutes<br><br>
📈 <strong>Forecast:</strong><br>
• Average delay: {pred['predicted_avg_delay']:.2f} minutes<br>
• Median delay: {pred['predicted_median']:.2f} minutes<br>
• Period: {pred['forecast_period']}"""
            return html
        except Exception as e:
            logger.exception("Prediction error")
            return "❌ I encountered an error generating predictions. Please ensure your data has the required columns (route, warehouse, delay_minutes) and try again."

    def get_recommendations(self) -> str:
        """Generate and return HTML recommendations."""
        if self.analytics is None:
            return "❌ Please load data first"
        try:
            recs = self.analytics.generate_recommendations()
            if not recs:
                return "No recommendations available at this time."
            parts = ["<strong>💡 Recommendations</strong><br><br>Based on my analysis of your shipment data:<br><br>"]
            for i, r in enumerate(recs, 1):
                emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(r.get("priority"), "⚪")
                parts.append(f"<strong>{i}. {r['category']}</strong> {emoji}<br>{r['finding']}<br>→ {r['action']}<br><br>")
            html = ''.join(parts)
            return html
        except Exception as e:
            logger.exception("Recommendations error")
            return "❌ I encountered an error generating recommendations. Please make sure your data is properly loaded and try again."

    def export_chat_history(self) -> Optional[str]:
        """Export chat history to JSON file and return path."""
        if not self.chat_history:
            return None
        out_path = os.path.join("logs", f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
            logger.info("Exported chat history to %s", out_path)
            return out_path
        except Exception as e:
            logger.exception("Failed to export chat history")
            return None


# -----------------------
# Console entrypoint
# -----------------------
def main() -> None:
    """Console entrypoint used by the console script and python -m hermes.app."""
    # Delayed import of UI to avoid heavy deps on import
    from .ui import create_gradio_app

    demo = create_gradio_app()
    # Launch - the UI code controls server options
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()