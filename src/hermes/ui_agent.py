"""
Gradio 5.48.0 Chat Interface with Agent Reasoning Visibility - TYPE-SAFE VERSION

This UI properly handles typed BaseResponse objects from the updated HermesApp.
"""

import gradio as gr
from gradio import ChatMessage
from typing import Iterator, Optional, List, Dict, Any, Union
import json
from datetime import datetime
import logging
import os
from pathlib import Path
import traceback

# Import app types
from .models import (
    BaseResponse,
    PredictionResponse,
    RecommendationResponse,
    VisualizationResponse,
    StatisticsResponse,
    GeneralResponse,
)

logger = logging.getLogger(__name__)


class HermesAgentTools:
    """
    Agent tools that work with typed BaseResponse objects.
    Tracks reasoning steps and classification decisions.
    """

    def __init__(self, hermes_app):
        """
        Args:
            hermes_app: HermesApp instance (with integrated type-safe handlers)
        """
        self.app = hermes_app
        self.reasoning_steps = []

    def classify_query_tool(self, query: str) -> dict:
        """
        Tool: Classify the user's query intent.

        Args:
            query: User's input query

        Returns:
            Dict with intent, confidence, and reasoning
        """
        self.reasoning_steps.append(
            {
                "step": "classify_query",
                "input": query[:100],
                "timestamp": datetime.now().isoformat(),
            }
        )

        try:
            classification = self.app.router.classify_query(query, smart_df=self.app.smart_df)
            intent = classification.get("intent", "general")
            confidence = classification.get("confidence", 0.0)

            result = {
                "intent": intent,
                "confidence": float(confidence),
                "reasoning": f"Query classified as **{intent}** with {confidence:.0%} confidence.",
            }
            self.reasoning_steps[-1]["output"] = result
            logger.info(f"Classification: {intent} ({confidence:.0%})")
            return result
        except Exception as e:
            traceback.print_exc()
            error_result = {
                "intent": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification failed: {str(e)}",
            }
            self.reasoning_steps[-1]["output"] = error_result
            logger.error(f"Classification error: {e}")
            return error_result

    def analyze_query(self, query: str, intent: str) -> Dict[str, Any]:
        """
        Process query with type-safe response model.
        
        Returns:
            Dict with 'success', 'response_model', and 'reasoning'
        """
        self.reasoning_steps.append({
            "step": "analyze_query",
            "input": f"{intent}: {query[:80]}",
            "timestamp": datetime.now().isoformat(),
        })
        
        try:
            # Get typed response from app
            response_model: BaseResponse = self.app.process_query_chat(query)
            
            # Handle both typed and dict responses (backward compatibility)
            if isinstance(response_model, BaseResponse):
                # Check if response actually succeeded
                response_success = getattr(response_model, 'success', True)
                error_msg = getattr(response_model, 'error', 'unknown error')
                reasoning_text = f"Processed {intent} query successfully" if response_success else f"(warning: {error_msg})"
                
                result = {
                    "success": response_success,
                    "response_model": response_model,
                    "reasoning": reasoning_text,
                }
                
                self.reasoning_steps[-1]["output"] = {
                    "success": response_success,
                    "intent": response_model.intent,
                    "has_chart": response_model.chart is not None if hasattr(response_model, 'chart') else False,
                }
            else:
                # Fallback for dict response
                result = {
                    "success": response_model.get("success", True),
                    "response_model": response_model,
                    "reasoning": f"Processed {intent} query (dict format)",
                }
                self.reasoning_steps[-1]["output"] = result
            
            return result
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Query processing error: {e}", exc_info=True)
            error_result = {
                "success": False,
                "response_model": None,
                "reasoning": f"Processing failed: {str(e)}",
            }
            self.reasoning_steps[-1]["output"] = error_result
            return error_result
    
    def get_reasoning_summary(self) -> str:
        """Format reasoning steps as markdown."""
        if not self.reasoning_steps:
            return "### 🔍 Agent Reasoning\n*Thinking...*"
        
        summary = "### 🔍 Agent Reasoning Chain\n\n"
        
        for i, step in enumerate(self.reasoning_steps, 1):
            step_name = step["step"].replace("_", " ").title()
            summary += f"**{i}. {step_name}**\n"
            summary += f"   - Input: `{step['input']}`\n"
            
            if "output" in step:
                output = step["output"]
                if isinstance(output, dict) and "reasoning" in output:
                    summary += f"   - Result: {output['reasoning']}\n"
                else:
                    summary += f"   - Result: ✅ Completed\n"
            summary += "\n"
        
        return summary
    
    def clear_reasoning(self):
        """Clear reasoning history."""
        self.reasoning_steps = []


def format_response_for_chat(response: Union[BaseResponse, Dict[str, Any]]) -> tuple[str, Optional[str], Optional[List], Optional[str]]:
    """
    Format typed response for Gradio display.
    
    Args:
        response: BaseResponse object or dict (legacy fallback)
    
    Returns:
        Tuple of (text, chart_path, dataframe_data, metrics_text)
    """
    try:
        # Handle typed response
        if isinstance(response, BaseResponse):
            text = response.text or ""
            chart_path = None
            dataframe_data = None
            metrics_text = None
            
            # Safe chart extraction
            try:
                if hasattr(response, 'chart') and response.chart:
                    if hasattr(response.chart, 'path'):
                        chart_path = response.chart.path
                    elif hasattr(response.chart, 'exists'):
                        # If chart object has exists() method, check it
                        if response.chart.exists():
                            chart_path = str(response.chart)
            except Exception as e:
                logger.warning(f"Error extracting chart: {e}")
            
            # Handle specific response types
            if isinstance(response, PredictionResponse):
                if response.metrics:
                    metrics_text = f"""
| Metric | Value |
|--------|-------|
| R² Score | {response.metrics.r2_score:.3f} |
| RMSE | {response.metrics.rmse:.2f} min |
| Model | {response.metrics.model_type} |
"""
            
            elif isinstance(response, RecommendationResponse):
                if response.recommendations:
                    priority_emoji = {
                        "Critical": "🔴",
                        "High": "🟠",
                        "Medium": "🟡",
                        "Low": "🟢"
                    }
                    recs_md = "\n\n**📋 Recommendations:**\n\n"
                    for i, rec in enumerate(response.recommendations, 1):
                        emoji = priority_emoji.get(rec.priority, "⚪")
                        recs_md += f"**{i}. {rec.category}** {emoji}\n"
                        recs_md += f"   - *Finding:* {rec.finding}\n"
                        recs_md += f"   - *Action:* {rec.action}\n\n"
                    text += recs_md
            
            elif isinstance(response, StatisticsResponse):
                if response.stats:
                    text += f"\n\n**📊 Dataset Overview:**\n"
                    text += f"- Date Range: {response.stats.date_range or 'N/A'}\n"
            
            elif isinstance(response, GeneralResponse):
                # Handle dataframe results
                if hasattr(response, 'data_type') and response.data_type == "dataframe":
                    if hasattr(response, 'raw_result') and response.raw_result is not None:
                        try:
                            import pandas as pd
                            if isinstance(response.raw_result, pd.DataFrame):
                                df = response.raw_result
                                dataframe_data = df.head(20).values.tolist()
                                dataframe_data.insert(0, df.columns.tolist())  # Add headers
                        except Exception as e:
                            logger.warning(f"Failed to format dataframe: {e}")
            
            return text, chart_path, dataframe_data, metrics_text
        
        # Fallback for dict response (legacy compatibility)
        else:
            text = response.get("text", "")
            chart_path = response.get("chart")
            dataframe_data = None
            metrics_text = None
            return text, chart_path, dataframe_data, metrics_text
    
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error formatting response: {e}", exc_info=True)
        # Return safe defaults
        return f"Error formatting response: {str(e)}", None, None, None


def create_agent_chat_interface(hermes_app) -> gr.Blocks:
    """
    Create Gradio 5.48.0 chat interface with agent reasoning display.
    Works with type-safe BaseResponse objects.

    Args:
        hermes_app: HermesApp instance (with integrated typed handlers)

    Returns:
        Gradio Blocks interface
    """

    tools = HermesAgentTools(hermes_app)

    custom_css = """
    .reasoning-panel { 
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
        border-left: 4px solid #0066cc; 
        padding: 16px; 
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        max-height: 650px;
        overflow-y: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .chat-container {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background: white;
    }
    .header {
        text-align: center;
        padding: 16px;
        background: linear-gradient(135deg, #0066cc 0%, #004080 100%);
        color: white;
        border-radius: 8px 8px 0 0;
        margin-bottom: 16px;
    }
    .metrics-panel {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 6px;
        margin-top: 8px;
    }
    """

    with gr.Blocks(
        title="Hermes Analytics Agent",
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", radius_size="lg"),
    ) as demo:
        # Header
        gr.Markdown(
            """
        # 🚀 Hermes Logistics Analytics Agent
        
        Ask natural language questions about shipments and logistics data. 
        Watch the agent reason through your query step-by-step.
        """,
            elem_classes="header"
        )

        # Hidden state for data loading
        data_loaded_state = gr.State(False)

        with gr.Row():
            with gr.Column(scale=7):
                # Main chatbot
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    type="messages",
                    height=650,
                    avatar_images=(
                        "https://api.dicebear.com/7.x/personas/svg?seed=user",
                        "https://api.dicebear.com/7.x/bottts/svg?seed=hermes",
                    ),
                    elem_classes="chat-container",
                )

            with gr.Column(scale=3):
                # Reasoning display
                reasoning_display = gr.Markdown(
                    "### 🔍 Agent Reasoning\n*Reasoning steps will appear here.*",
                    elem_classes="reasoning-panel",
                    label="Reasoning Chain",
                )
                
                # Metrics display
                metrics_display = gr.Markdown(
                    "",
                    label="📈 Metrics",
                    visible=False,
                    elem_classes="metrics-panel",
                )

        # DataFrame output (collapsible)
        dataframe_output = gr.Dataframe(
            label="📋 Data Preview",
            visible=False,
            interactive=False,
            wrap=True,
        )

        # Input area - unified query box with data selector
        with gr.Row(elem_classes="input-row"):
            data_selector = gr.Dropdown(
                choices=hermes_app.get_csv_files(),
                label="",
                value=hermes_app.get_csv_files()[0] if hermes_app.get_csv_files() else None,
                info="Select data",
                scale=1,
                show_label=False,
            )
            msg_input = gr.Textbox(
                placeholder="Ask about shipments, delays, trends... 📊",
                scale=4,
                lines=1,
                max_lines=3,
                show_label=False,
            )
            submit_btn = gr.Button("🚀 Send", scale=1, variant="primary", size="lg")

        # Quick suggestions
        gr.Markdown("**💡 Try asking:**")
        
        with gr.Row():
            suggestion_1 = gr.Button(
                "📈 Show delivery trends by route",
                scale=1,
                size="sm",
                variant="secondary",
            )
            suggestion_2 = gr.Button(
                "⏱️ Which routes have most delays?",
                scale=1,
                size="sm",
                variant="secondary",
            )
            suggestion_3 = gr.Button(
                "🏢 Compare warehouse performance",
                scale=1,
                size="sm",
                variant="secondary",
            )

        # Utility buttons
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", scale=1, size="sm", variant="secondary")
            export_btn = gr.Button("📥 Export History", scale=1, size="sm", variant="secondary")

        # =====================================================================
        # Event Handlers
        # =====================================================================
        
        def clear_conversation():
            tools.clear_reasoning()
            return (
                [],  # chatbot
                "### 🔍 Agent Reasoning\n*Reasoning steps will appear here.*",  # reasoning
                "",  # metrics
                None,  # dataframe
                gr.update(visible=False),  # metrics visibility
                gr.update(visible=False),  # dataframe visibility
            )

        clear_btn.click(
            clear_conversation, 
            outputs=[chatbot, reasoning_display, metrics_display, dataframe_output, metrics_display, dataframe_output]
        )

        # Main response handler
        def respond(
            message: str,
            history: List,
            loaded_state: bool,
            selected_data: Optional[str]
        ) -> Iterator:
            """
            Main chat handler with typed responses.

            Yields progressively:
            1. Auto-load data if needed (silently, on first query)
            2. User message added
            3. Classification thinking
            4. Query analysis
            5. Final result with chart (if applicable)
            """

            if not message.strip():
                return

            # Step 0: Auto-load data on first query if not already loaded
            if not loaded_state or hermes_app.current_df is None or hermes_app.smart_df is None:
                try:
                    # Load default data silently
                    df, load_msg = hermes_app.load_data("Select Existing", None, selected_data)
                    
                    if df is None:
                        # Only show error if data loading explicitly failed
                        history.append(ChatMessage(
                            role="assistant",
                            content=f"❌ **Unable to load data**\n\n{load_msg}\n\nPlease check that a CSV file exists in the data directory.",
                            metadata={"title": "⚠️ Data Error"}
                        ))
                        yield (
                            history,
                            "❌ Data loading failed",
                            "",
                            None,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            loaded_state
                        )
                        return
                    
                    # Data loaded successfully - continue silently
                    loaded_state = True
                    logger.info(f"✅ Data auto-loaded: {load_msg}")
                    
                except Exception as e:
                    logger.error(f"Auto-load error: {e}")
                    history.append(ChatMessage(
                        role="assistant",
                        content=f"❌ **Error loading data:** {str(e)}",
                        metadata={"title": "⚠️ Error"}
                    ))
                    yield (
                        history,
                        "❌ Data loading failed",
                        "",
                        None,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        loaded_state
                    )
                    return

            # Step 1: Add user message
            history.append(ChatMessage(role="user", content=message))
            yield (
                history,
                "🤔 *Analyzing your query...*",
                "",
                None,
                gr.update(visible=False),
                gr.update(visible=False),
                loaded_state
            )

            # Step 2: Classify query
            try:
                classify_result = tools.classify_query_tool(message)
                detected_intent = classify_result.get("intent", "unknown")
                reasoning_md = tools.get_reasoning_summary()

                # Show thinking message
                history.append(
                    ChatMessage(
                        role="assistant",
                        content=f"🤔 **Analyzing your query...**\n\n{reasoning_md}",
                        metadata={"title": "🔍 Thinking"},
                    )
                )
                yield (
                    history,
                    reasoning_md,
                    "",
                    None,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    loaded_state
                )

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Classification error: {e}")
                history.append(
                    ChatMessage(
                        role="assistant",
                        content=f"❌ **Error during classification:** {str(e)}",
                    )
                )
                yield (
                    history,
                    f"❌ Error: {str(e)}",
                    "",
                    None,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    loaded_state
                )
                return

            # Step 3: Process query with typed response
            try:
                result = tools.analyze_query(message, detected_intent)
                reasoning_md = tools.get_reasoning_summary()
                if not result["success"]:
                    raise Exception(result["reasoning"])

                response_obj = result["response_model"]
                
                if response_obj is None:
                    raise Exception("No response object returned")

                # Remove the "thinking" message
                if history and history[-1].metadata.get("title") == "🔍 Thinking":
                    history.pop()

                # Format response for display
                try:
                    text, chart_path, df_data, metrics_text = format_response_for_chat(response_obj)
                except Exception as format_err:
                    traceback.print_exc()
                    logger.error(f"Format response error: {format_err}", exc_info=True)
                    text = f"Response formatted successfully but display error: {str(format_err)}"
                    chart_path = None
                    df_data = None
                    metrics_text = None

                # Add text response
                history.append(
                    ChatMessage(
                        role="assistant",
                        content=text,
                        metadata={"title": "📊 Results"},
                    )
                )

                # Add chart if present
                if chart_path and isinstance(chart_path, str) and os.path.exists(chart_path):
                    try:
                        history.append(
                            ChatMessage(
                                role="assistant",
                                content={"path": chart_path, "mime_type": "image/png"},
                            )
                        )
                        logger.info(f"Added chart to message: {chart_path}")
                    except Exception as chart_err:
                        traceback.print_exc()
                        logger.warning(f"Failed to add chart: {chart_err}")

                # Update reasoning
                tools.clear_reasoning()
                
                # Get intent from response
                try:
                    intent_str = response_obj.intent if isinstance(response_obj, BaseResponse) else detected_intent
                    success_str = response_obj.success if isinstance(response_obj, BaseResponse) else True
                    
                    # Safe timestamp extraction
                    if isinstance(response_obj, BaseResponse) and hasattr(response_obj, 'timestamp'):
                        timestamp_str = response_obj.timestamp.strftime('%H:%M:%S')
                    else:
                        timestamp_str = datetime.now().strftime('%H:%M:%S')
                except Exception as attr_err:
                    traceback.print_exc()
                    logger.warning(f"Error extracting response attributes: {attr_err}")
                    intent_str = detected_intent
                    success_str = True
                    timestamp_str = datetime.now().strftime('%H:%M:%S')
                
                final_reasoning = (
                    f"✅ **Query Complete**\n\n"
                    f"**Intent:** {intent_str}\n"
                    f"**Success:** {success_str}\n"
                    f"**Timestamp:** {timestamp_str}\n"
                )

                # Show outputs
                metrics_visible = metrics_text is not None and len(metrics_text) > 0
                df_visible = df_data is not None

                yield (
                    history,
                    final_reasoning,
                    metrics_text or "",
                    df_data,
                    gr.update(visible=metrics_visible),
                    gr.update(visible=df_visible),
                    loaded_state
                )

            except Exception as e:
                traceback.print_exc()
                logger.exception(f"Query processing error: {e}")
                error_msg = f"❌ **Error processing query:** {str(e)}"
                
                # Remove thinking message if it's still there
                if history and history[-1].metadata.get("title") == "🔍 Thinking":
                    history.pop()
                
                history.append(
                    ChatMessage(
                        role="assistant",
                        content=error_msg,
                    )
                )
                yield (
                    history,
                    f"❌ Error: {str(e)}",
                    "",
                    None,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    loaded_state
                )

        # Connect events
        outputs = [
            chatbot,
            reasoning_display,
            metrics_display,
            dataframe_output,
            metrics_display,  # visibility
            dataframe_output,  # visibility
            data_loaded_state,
        ]

        msg_input.submit(
            respond, [msg_input, chatbot, data_loaded_state, data_selector], outputs
        ).then(lambda: "", None, msg_input)
        
        submit_btn.click(
            respond, [msg_input, chatbot, data_loaded_state, data_selector], outputs
        ).then(lambda: "", None, msg_input)

        # Suggestion buttons
        def handle_suggestion(btn_label):
            return btn_label
        
        suggestion_1.click(
            handle_suggestion, suggestion_1, msg_input
        ).then(
            respond, [msg_input, chatbot, data_loaded_state, data_selector], outputs
        ).then(lambda: "", None, msg_input)
        
        suggestion_2.click(
            handle_suggestion, suggestion_2, msg_input
        ).then(
            respond, [msg_input, chatbot, data_loaded_state, data_selector], outputs
        ).then(lambda: "", None, msg_input)
        
        suggestion_3.click(
            handle_suggestion, suggestion_3, msg_input
        ).then(
            respond, [msg_input, chatbot, data_loaded_state, data_selector], outputs
        ).then(lambda: "", None, msg_input)

    return demo