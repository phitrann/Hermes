"""
Application module for the hermes package.

Defines HermesApp (the application object used by UI) and main()
(entrypoint used by the console script `hermes`).

This module ties together router, analytics, visualizer, and semantic helpers.
It uses the PandasAI Dataframe for LLM-driven queries and preserves
the same handlers used by the earlier single-file implementation.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union, Literal

import pandas as pd
import pandasai as pai
from pandasai.core.response import DataFrameResponse, NumberResponse, ChartResponse, StringResponse
    
from .config import CHARTS_DIR, DATA_DIR, QUESTIONS_FILE, SHIPMENTS_FILE, llm
from .analytics import HermesAnalytics
from .visualizer import HermesVisualizer
from .router import QueryRouter
from .semantic import register_semantic_dataset
from .autoviz import auto_visualize

from .models import (
    BaseResponse,
    PredictionResponse,
    RecommendationResponse,
    VisualizationResponse,
    StatisticsResponse,
    GeneralResponse,
    ChartData,
    MetricsData,
    PredictionData,
    RecommendationItem,
    StatsSummary,
    DataFramePreview,
    ResponseFactory,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

pai.config.set({"llm": llm, "enable_cache": True})


class HermesApp:
    """Main application class providing data loading, query processing and helpers."""

    def __init__(self) -> None:
        self.chat_history: list[Dict[str, Any]] = []
        self.current_df: Optional[pd.DataFrame] = None
        self.smart_df: Optional[pai.DataFrame] = None
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
            self.smart_df = pai.DataFrame(df)
            self.analytics = HermesAnalytics(df)

            logger.info("Loaded %d records from %s", len(df), data_path)
            return df, f"✅ Loaded {len(df)} shipment records"
        except Exception as e:
            logger.exception("Data loading error")
            return None, f"❌ Error loading data: {e}"

    # -----------------------
    # Query processing with TYPE-SAFE RESPONSES
    # -----------------------
    def process_query_chat(self, user_prompt: str) -> Union[Dict[str, Any], BaseResponse]:
        """
        Chat-optimized query processing function with TYPE-SAFE responses.
        
        Returns:
            BaseResponse subclass (typed) or dict (legacy fallback)
        """
        if not user_prompt or not user_prompt.strip():
            return GeneralResponse(
                text="❌ Please enter a question.",
                intent="general",
                success=False,  
                error="Empty query"
            )
        
        # Ensure data is loaded
        if self.current_df is None or self.smart_df is None:
            return GeneralResponse(
                text="❌ Please load data first using the 'Load Data' button in the sidebar.",
                intent="general",
                success=False,
                error="Data not loaded"
            )
        
        prompt = user_prompt.strip()
        
        try:
            # LLM-driven routing
            classification = self.router.classify_query(prompt, smart_df=self.smart_df)
            intent = classification.get("intent", "general")
            confidence = classification.get("confidence", 0.0)
            method = classification.get("method", "fallback")
            force_chart = self.router.should_force_chart(prompt, smart_df=self.smart_df)
            
            logger.info("Routing: intent=%s confidence=%.2f method=%s force_chart=%s", intent, confidence, method, force_chart)
            
            # Dispatch to TYPE-SAFE handlers
            if intent == "prediction":
                result = self._handle_prediction_chat(prompt)
            elif intent == "recommendation":
                result = self._handle_recommendation_chat(prompt)
            elif intent in ("visualization", "comparison"):
                result = self._handle_visualization_chat(prompt, force_chart)
            elif intent == "statistics":
                result = self._handle_stats_chat(prompt)
            else:
                result = self._handle_general_chat(prompt, force_chart)
            
            logger.info(f"{result = }")
            # Add to chat history (convert to dict for storage)
            self.chat_history.append({
                "query": prompt,
                "response": result.text if isinstance(result, BaseResponse) else result.get("text", ""),
                "intent": result.intent if isinstance(result, BaseResponse) else intent,
                "chart_generated": (result.chart is not None and result.chart.exists) if isinstance(result, BaseResponse) else result.get("chart") is not None,
                "chart_path": result.chart.path if isinstance(result, BaseResponse) and getattr(result, "chart", None) else None,
                "timestamp": result.timestamp.isoformat() if isinstance(result, BaseResponse) else datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.exception("Query processing error")
            return GeneralResponse(
                text=f"❌ **Error processing query:** {str(e)}\n\nPlease try rephrasing your question or check the data.",
                intent="general",
                success=False,
                error=str(e),
                metadata={"error": str(e)}
            )

    # -----------------------
    # TYPE-SAFE Chat Handlers (NEW)
    # -----------------------
    def _handle_prediction_chat(self, prompt: str) -> PredictionResponse:
        """Handle prediction with validated response model."""
        logger.info("Handling prediction request (typed)")
        
        try:
            metrics_dict = self.analytics.train_prediction_model()
            pred_dict = self.analytics.predict_next_week()
            
            if not metrics_dict or not pred_dict:
                return PredictionResponse(
                    text="❌ Unable to generate predictions. Ensure data has required columns (date, delay_minutes, route, warehouse).",
                    intent="prediction",
                    success=False,
                    error="Missing required columns or insufficient data"
                )
            
            # Validate and structure data
            metrics = MetricsData(**metrics_dict)
            prediction = PredictionData(**pred_dict)
            
            # Format conversational response
            response_text = f"""🔮 **Delay Prediction Forecast**

**Model Performance:**
- R² Score: {metrics.r2_score:.3f}
- RMSE: {metrics.rmse:.2f} minutes
- Model: {metrics.model_type}

**Next Week Forecast:**
- Average Delay: **{prediction.predicted_avg_delay:.2f} minutes**
- Median Delay: {prediction.predicted_median:.2f} minutes
- Period: {prediction.forecast_period}

Based on historical patterns, you can expect moderate delays. Consider reviewing routes with consistently high delays.
"""
            
            return PredictionResponse(
                text=response_text,
                intent="prediction",
                metrics=metrics,
                prediction=prediction,
                metadata={
                    "metrics": metrics.dict(),
                    "prediction": prediction.dict()
                }
            )
            
        except Exception as e:
            logger.exception("Prediction error")
            return PredictionResponse(
                text=f"❌ **Prediction Error:** {str(e)}\n\nMake sure your data includes columns: date, delay_minutes, route, and warehouse.",
                intent="prediction",
                success=False,
                error=str(e),
                metadata={"error": str(e)}
            )

    def _handle_recommendation_chat(self, prompt: str) -> RecommendationResponse:
        """Handle recommendations with validated response model."""
        logger.info("Handling recommendation request (typed)")
        
        try:
            recs_dicts = self.analytics.generate_recommendations()
            
            if not recs_dicts:
                return RecommendationResponse(
                    text="ℹ️ No specific recommendations available at this time. Your logistics operations appear to be running smoothly!",
                    intent="recommendation",
                    recommendations=[]
                )
            
            # Validate recommendations
            recommendations = [RecommendationItem(**r) for r in recs_dicts]
            
            # Format response
            priority_emoji = {
                "Critical": "🔴",
                "High": "🟠",
                "Medium": "🟡",
                "Low": "🟢"
            }
            
            response_text = "💡 **Recommendations to Improve Your Logistics**\n\n"
            for i, rec in enumerate(recommendations, 1):
                emoji = priority_emoji.get(rec.priority, "⚪")
                response_text += f"**{i}. {rec.category}** {emoji}\n"
                response_text += f"*Finding:* {rec.finding}\n"
                response_text += f"*Action:* {rec.action}\n\n"
            
            return RecommendationResponse(
                text=response_text,
                intent="recommendation",
                recommendations=recommendations,
                metadata={"recommendations": [r.dict() for r in recommendations]}
            )
            
        except Exception as e:
            logger.exception("Recommendations error")
            return RecommendationResponse(
                text=f"❌ **Error generating recommendations:** {str(e)}",
                intent="recommendation",
                success=False,
                error=str(e),
                recommendations=[]
            )

    def _handle_visualization_chat(self, prompt: str, force_chart: bool = True) -> VisualizationResponse:
        """Handle visualization with validated response model."""
        logger.info("Handling visualization request (typed)")
        
        try:
            # Get time context
            max_date = self._get_max_date_str()
            
            # Request visualization from LLM
            viz_prompt = f"<TIME_CONTEXT>Current date: {max_date}</TIME_CONTEXT>\n{prompt}\nCreate and save a visualization as PNG. /no_think"
            response = self.smart_df.chat(viz_prompt)
            
            # Get generated chart
            chart_path = self.visualizer.get_latest_chart()
            
            chart_data = None
            if chart_path:
                chart_data = ChartData(
                    path=chart_path,
                    mime_type="image/png",
                    caption=f"Visualization for: {prompt[:100]}"
                )
            
            # Format response text
            response_text = str(response) if response else "✅ Visualization created!"
            if not chart_data:
                response_text += "\n\n⚠️ *Chart generation was requested but none produced. The LLM may have provided a text response instead.*"
            
            return VisualizationResponse(
                text=f"📊 {response_text}",
                chart=chart_data,
                intent="visualization",
                chart_type=self._infer_chart_type(prompt),
                metadata={
                    "chart_path": chart_path,
                    "chart_type": self._infer_chart_type(prompt)
                }
            )
            
        except Exception as e:
            logger.exception("Visualization error")
            return VisualizationResponse(
                text=f"❌ **Visualization Error:** {str(e)}\n\nTry rephrasing your request or ask for a specific type of chart.",
                intent="visualization",
                success=False,
                error=str(e)
            )

    def _handle_stats_chat(self, prompt: str) -> StatisticsResponse:
        """Handle statistics with validated response model."""
        logger.info("Handling statistics request (typed)")
        
        try:
            max_date = self._get_max_date_str()
            full_prompt = f"<TIME_CONTEXT>Current date: {max_date}</TIME_CONTEXT>\n{prompt} /no_think"
            
            response = self.smart_df.chat(full_prompt)
            stats_dict = self.analytics.get_summary_stats() if self.analytics else {}
            
            # Validate stats
            stats = StatsSummary(**stats_dict) if stats_dict else None
            
            # Format response
            response_text = f"📊 **Statistics Summary**\n\n{str(response)}"
            
            if stats:
                response_text += f"\n\n**Quick Stats:**\n"
                response_text += f"- Total Shipments: {stats.total_shipments:,}\n"
                response_text += f"- Delayed Shipments: {stats.delayed_shipments:,}\n"
                response_text += f"- Average Delay: {stats.avg_delay_minutes:.2f} min\n"
                response_text += f"- Delay Rate: {stats.delay_rate:.1%}\n"
            
            return StatisticsResponse(
                text=response_text,
                intent="statistics",
                stats=stats,
                metadata=stats.dict() if stats else {}
            )
            
        except Exception as e:
            logger.exception("Statistics error")
            return StatisticsResponse(
                text=f"❌ **Statistics Error:** {str(e)}",
                intent="statistics",
                success=False,
                error=str(e)
            )

    def _handle_general_chat(self, prompt: str, force_chart: bool = False) -> GeneralResponse:
        """Handle general queries with validated response model."""
        logger.info("Handling general query (typed)")
        
        try:
            max_date = self._get_max_date_str()
            full_prompt = f"<TIME_CONTEXT>Current date: {max_date}</TIME_CONTEXT>\n{prompt} /no_think"
            if force_chart:
                full_prompt += "\nInclude a relevant visualization."
            
            response = self.smart_df.chat(full_prompt)
            
            # Determine data type
            data_type = "text"
            if isinstance(response, DataFrameResponse):
                data_type = "dataframe"
            elif isinstance(response, NumberResponse):
                data_type = "number"
            elif isinstance(response, ChartResponse):
                data_type = "chart"
            
            response_text = f"{str(response)}"
            
            return GeneralResponse(
                text=response_text,
                intent="general",
                data_type=data_type,
                raw_result=response,
                metadata={
                    "data_type": data_type,
                    "raw": response
                }
            )
            
        except Exception as e:
            logger.exception("General query error")
            return GeneralResponse(
                text=f"❌ **Error:** {str(e)}\n\nI encountered an issue processing your request. Please try again or rephrase your question.",
                intent="general",
                success=False,
                error=str(e)
            )

    # -----------------------
    # Helper Methods
    # -----------------------
    def _get_max_date_str(self) -> str:
        """Get max date from dataframe as string."""
        if self.current_df is None or "date" not in self.current_df.columns:
            return ""
        try:
            return self.current_df["date"].max().strftime("%Y-%m-%d")
        except Exception:
            return ""
    
    def _infer_chart_type(self, prompt: str) -> str:
        """Infer chart type from prompt keywords."""
        prompt_lower = prompt.lower()
        
        if any(kw in prompt_lower for kw in ["trend", "over time", "timeline"]):
            return "line"
        elif any(kw in prompt_lower for kw in ["compare", "comparison", "vs"]):
            return "bar"
        elif any(kw in prompt_lower for kw in ["distribution", "histogram"]):
            return "histogram"
        elif any(kw in prompt_lower for kw in ["scatter", "correlation"]):
            return "scatter"
        elif any(kw in prompt_lower for kw in ["pie", "proportion", "share"]):
            return "pie"
        else:
            return "unknown"

    # -----------------------
    # LEGACY HANDLERS (kept for backward compatibility)
    # -----------------------
    def process_query(self, data_source_type: str, uploaded_file: Optional[Any], selected_file: Optional[str],
                      selected_question: Optional[str], user_prompt: Optional[str]) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Optional[pd.DataFrame], list[Dict[str, Any]]]:
        """
        Legacy query processing function (kept for backward compatibility).

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
            # Get typed response
            response = self.process_query_chat(prompt)
            
            # Convert typed response to legacy format
            if isinstance(response, BaseResponse):
                formatted_html = self._beautify_response(response.text, 
                                                        title=f"🤖 Hermes {response.intent.title()} Response",
                                                        footer="Powered by PandasAI & Local LLM")
                chart_path = response.chart.path if response.chart else None
                stats = response.metadata.get("stats", response.metadata)
                preview = self.current_df.head(10) if self.current_df is not None else None
                
                return formatted_html, chart_path, stats, preview, self.chat_history
            else:
                # Fallback for dict response
                return self._beautify_response(str(response.get("text", ""))), response.get("chart"), response.get("metadata", {}), None, self.chat_history

        except Exception as e:
            logger.exception("Query processing error")
            err = f"<div style='color:red;'><strong>Error processing query:</strong> {e}</div>"
            return err, None, None, None, self.chat_history

    def _beautify_response(self, response_obj: Any, title: str = "Hermes AI Response", footer: str = "") -> str:
        """Return HTML-wrapped response string for display in UI."""
        # Convert response to safe string and replace newlines with <br>
        response_str = str(response_obj).replace("\n", "<br>")
        html = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 12px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="font-size:1.6em;">🤖</span>
    <strong style="font-size:1.05em;">{title}</strong>
  </div>
  <div style="margin-top:10px; background: rgba(255,255,255,0.06); padding:12px; border-radius:8px;">{response_str}</div>
  <div style="text-align:right; margin-top:8px; font-size:0.85em; opacity:0.9;">{footer}</div>
</div>
"""
        return html

    def get_predictions(self) -> str:
        """Train model and return formatted prediction results (LEGACY HTML FORMAT)."""
        if self.analytics is None:
            return "❌ Please load data first"
        try:
            response = self._handle_prediction_chat("prediction")
            if isinstance(response, PredictionResponse) and response.success:
                html = f"""
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 14px; border-radius: 10px;">
  <strong>🔮 Prediction</strong>
  <div style="margin-top:8px; background: rgba(255,255,255,0.04); padding:10px; border-radius:8px;">
    <strong>Model:</strong> R² {response.metrics.r2_score:.3f} • RMSE {response.metrics.rmse:.2f} min<br>
    <strong>Forecast:</strong> Avg delay {response.prediction.predicted_avg_delay:.2f} min (median {response.prediction.predicted_median:.2f})<br>
    <em>Period:</em> {response.prediction.forecast_period}
  </div>
</div>
"""
                return html
            else:
                return f"❌ {response.error or 'Prediction failed'}"
        except Exception as e:
            logger.exception("Prediction error")
            return f"❌ Prediction error: {e}"

    def get_recommendations(self) -> str:
        """Generate and return HTML recommendations (LEGACY HTML FORMAT)."""
        if self.analytics is None:
            return "❌ Please load data first"
        try:
            response = self._handle_recommendation_chat("recommendations")
            if isinstance(response, RecommendationResponse) and response.recommendations:
                parts = []
                for i, r in enumerate(response.recommendations, 1):
                    emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(r.priority, "⚪")
                    parts.append(f"<div style='padding:8px;margin:8px 0;border-left:4px solid rgba(255,255,255,0.12);'><strong>{i}. {r.category}</strong> {emoji}<br><small>{r.finding}</small><br>{r.action}</div>")
                html = f"""
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 14px; border-radius: 10px;">
  <strong>💡 Recommendations</strong>
  <div style="margin-top:8px; background: rgba(255,255,255,0.03); padding:10px; border-radius:8px;">
    {''.join(parts)}
  </div>
</div>
"""
                return html
            else:
                return "ℹ️ No recommendations available"
        except Exception as e:
            logger.exception("Recommendations error")
            return f"❌ Recommendations error: {e}"

    def handle_query(self, query: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Optional[pd.DataFrame], list]:
        """
        Unified query handler for UI compatibility.
        Wraps process_query for backward compatibility.

        Args:
            query: User's natural language query

        Returns:
            Tuple of (formatted_html, chart_path, stats_dict, preview_df, chat_history)
        """
        if self.current_df is None or self.smart_df is None:
            return "❌ Please load data first", None, None, None, self.chat_history

        return self.process_query(
            data_source_type="select",
            uploaded_file=None,
            selected_file=None,
            selected_question=None,
            user_prompt=query
        )

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
    """Console entrypoint with agent reasoning UI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Logistics Analytics")
    parser.add_argument(
        "--ui",
        choices=["agent", "legacy"],
        default="agent",
        help="UI mode: agent (with reasoning) or legacy (original)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio server port",
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Hermes Analytics with %s UI", args.ui)
    
    # Initialize app
    hermes_app = HermesApp()
    
    if args.ui == "agent":
        # New agent reasoning interface with type-safe responses
        from .ui_agent import create_agent_chat_interface
        demo = create_agent_chat_interface(hermes_app)
    else:
        # Legacy UI
        from .ui import create_gradio_app
        demo = create_gradio_app()
    
    # Launch
    logger.info("🌐 Launching on http://0.0.0.0:%d", args.port)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()