"""
Gradio 5.48.0 Chat Interface with Intelligent Component Routing and Agent Tool Display

This UI automatically selects the best Gradio component for each response type:
- DataFrames → Inline in chat
- Numbers → Inline in chat with highlighting
- Charts → Inline in chat
- Text → Markdown in chat
- Agent reasoning → Step-by-step with metadata and timing
"""

import gradio as gr
from gradio import ChatMessage
from typing import Iterator, Optional, List, Dict, Any
import time
from datetime import datetime
import logging
import os
import json
import traceback

from pandasai.helpers.json_encoder import CustomJsonEncoder

# Import app types
from .models import BaseResponse

from .utils import (
    load_questions_dataset,
    get_random_suggestions,
    extract_response_components,
    LLMReasoningCapture
)

# Initialize the reasoning capture handler
reasoning_capture = LLMReasoningCapture()

# Attach to relevant loggers
for logger_name in ['hermes', 'pandasai', 'litellm']:
    logging.getLogger(logger_name).addHandler(reasoning_capture)

logger = logging.getLogger(__name__)


class HermesAgentTools:
    """Agent tools that work with typed BaseResponse objects."""

    def __init__(self, hermes_app):
        self.app = hermes_app
        self.reasoning_steps = []
        self.step_timings = {}

    def classify_query_tool(self, query: str) -> dict:
        """Tool: Classify the user's query intent."""
        start_time = time.time()
        
        self.reasoning_steps.append({
            "step": "classify_query",
            "input": query[:100],
            "timestamp": datetime.now().isoformat(),
        })

        try:
            classification = self.app.router.classify_query(query, smart_df=self.app.smart_df)
            intent = classification.get("intent", "general")
            confidence = classification.get("confidence", 0.0)

            duration = time.time() - start_time
            self.step_timings['classification'] = duration

            result = {
                "intent": intent,
                "confidence": float(confidence),
                "duration": duration,
                "reasoning": f"Query classified as **{intent}** with {confidence:.0%} confidence.",
            }
            self.reasoning_steps[-1]["output"] = result
            logger.info(f"Classification: {intent} ({confidence:.0%}) in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            traceback.print_exc()
            error_result = {
                "intent": "unknown",
                "confidence": 0.0,
                "duration": duration,
                "reasoning": f"Classification failed: {str(e)}",
            }
            self.reasoning_steps[-1]["output"] = error_result
            logger.error(f"Classification error: {e}")
            return error_result

    def analyze_query(self, query: str, intent: str) -> Dict[str, Any]:
        """Process query with type-safe response model."""
        start_time = time.time()
        
        self.reasoning_steps.append({
            "step": "analyze_query",
            "input": f"{intent}: {query[:80]}",
            "timestamp": datetime.now().isoformat(),
        })
        
        try:
            response_model: BaseResponse = self.app.process_query_chat(query)
            
            duration = time.time() - start_time
            self.step_timings['analysis'] = duration
            
            if isinstance(response_model, BaseResponse):
                response_success = getattr(response_model, 'success', True)
                error_msg = getattr(response_model, 'error', 'unknown error')
                reasoning_text = f"Processed {intent} query successfully" if response_success else f"(warning: {error_msg})"
                
                result = {
                    "success": response_success,
                    "response_model": response_model,
                    "reasoning": reasoning_text,
                    "duration": duration,
                }
                
                self.reasoning_steps[-1]["output"] = {
                    "success": response_success,
                    "intent": response_model.intent,
                    "has_chart": response_model.chart is not None if hasattr(response_model, 'chart') else False,
                }
            else:
                result = {
                    "success": response_model.get("success", True),
                    "response_model": response_model,
                    "reasoning": f"Processed {intent} query (dict format)",
                    "duration": duration,
                }
                self.reasoning_steps[-1]["output"] = result
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            traceback.print_exc()
            logger.error(f"Query processing error: {e}", exc_info=True)
            error_result = {
                "success": False,
                "response_model": None,
                "reasoning": f"Processing failed: {str(e)}",
                "duration": duration,
            }
            self.reasoning_steps[-1]["output"] = error_result
            return error_result
    
    def clear_reasoning(self):
        """Clear reasoning history."""
        self.reasoning_steps = []
        self.step_timings = {}


def create_agent_chat_interface(hermes_app) -> gr.Blocks:
    """Create Gradio interface with intelligent component routing and agent tool display."""

    tools = HermesAgentTools(hermes_app)

    custom_css = """
    .chat-container {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background: white;
    }
    .header {
        text-align: center;
        padding: 12px 16px;
        background: linear-gradient(135deg, #0066cc 0%, #004080 100%);
        color: white;
        border-radius: 8px 8px 0 0;
        margin-bottom: 12px;
    }
    .metrics-panel {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
        border: 1px solid #e0e0e0;
    }
    .input-section {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .suggestion-row {
        gap: 6px;
        margin-bottom: 8px;
    }
    .suggestion-btn {
        border-radius: 20px !important;
        padding: 6px 12px !important;
        font-size: 0.75em !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 100% !important;
        height: 36px !important;
        line-height: 1.2 !important;
    }
    /* Number result styling */
    .number-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
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
        Watch the agent reason through your query with full tool usage visibility.
        """,
            elem_classes="header"
        )

        # Load questions dataset
        from .config import DATA_DIR
        all_questions = load_questions_dataset(DATA_DIR)
        
        # Hidden states
        data_loaded_state = gr.State(False)
        questions_state = gr.State(all_questions)

        with gr.Row():
            with gr.Column(scale=7):
                # Status indicator
                with gr.Row():
                    status_text = gr.Markdown("**📊 Status:** Ready to analyze", elem_classes="status-indicator")
                
                # Main chatbot (without examples param)
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    type="messages",
                    height=500,
                    avatar_images=(
                        "https://api.dicebear.com/7.x/personas/svg?seed=user",
                        "https://api.dicebear.com/7.x/bottts/svg?seed=hermes",
                    ),
                    elem_classes="chat-container",
                    show_copy_button=True,
                    placeholder="👋 Hello! I'm your logistics analytics assistant. Ask me anything about your shipment data!",
                )
                
                # Input section with examples above it
                with gr.Group(elem_classes="input-section"):
                    # Example questions above input
                    gr.Markdown("**💡 Try asking:**")
                    with gr.Row(
                            elem_classes="suggestion-row"
                        ):
                        suggestion_btns = []
                        for i in range(3):
                            btn = gr.Button(
                                "Loading...",
                                scale=1,
                                size="sm",
                                variant="secondary",
                                elem_id=f"suggestion_{i}",
                                elem_classes="suggestion-btn",
                            )
                            suggestion_btns.append(btn)
                        
                        reload_btn = gr.Button(
                            "🔄",
                            scale=0.15,
                            size="sm",
                            variant="secondary",
                            elem_classes="suggestion-btn",
                        )
                    
                    # Input area
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask about shipments, delays, trends... 📊",
                            scale=5,
                            lines=1,
                            max_lines=3,
                            show_label=False,
                            container=False,
                        )
                        submit_btn = gr.Button("🚀 Send", scale=1, variant="primary", size="lg")

            with gr.Column(scale=3):
                # Data selector
                csv_files = hermes_app.get_csv_files()
                data_selector = gr.Dropdown(
                    choices=[(f.name, str(f)) for f in csv_files],
                    label="📁 Data Source",
                    value=str(csv_files[0]) if csv_files else None,
                )
                
                # Metrics display (for predictions)
                metrics_display = gr.JSON(
                    label="📈 Model Metrics",
                    visible=False,
                    elem_classes="metrics-panel",
                )
                
                # Stats display (for statistics queries)
                stats_display = gr.JSON(
                    label="📊 Statistics Summary",
                    visible=False,
                    elem_classes="metrics-panel",
                )

        # Utility buttons
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", scale=1, size="sm", variant="secondary")
            gr.Markdown("*💡 Tip: Click suggestion buttons above to get started quickly!*")

        # =====================================================================
        # Helper Functions
        # =====================================================================
        
        def yield_state(history, loaded_state, status_msg="Processing...", 
                       metrics=None, stats=None,
                       show_metrics=False, show_stats=False):
            """Helper to yield consistent state tuple."""
            return (
                history,
                status_msg,
                metrics,
                stats,
                gr.update(visible=show_metrics),
                gr.update(visible=show_stats),
                loaded_state
            )
        
        def add_assistant_message(history, content, metadata=None):
            """Helper to add assistant message to history."""
            history.append(ChatMessage(
                role="assistant",
                content=content,
                metadata=metadata,
            ))
            return history
        
        def add_error_message(history, error_msg):
            """Helper to add error message to history."""
            return add_assistant_message(
                history,
                f"❌ **Error:** {error_msg}"
            )
        
        # =====================================================================
        # Event Handlers
        # =====================================================================
        
        def reload_suggestions(questions: List[str]) -> tuple:
            """Generate new random suggestions."""
            try:
                new_suggestions = get_random_suggestions(questions, 3)
                while len(new_suggestions) < 3:
                    new_suggestions.append("No more questions available")
                
                return tuple(new_suggestions)
            except Exception as e:
                logger.error(f"Error reloading suggestions: {e}")
                return ("Error", "Error", "Error")
        
        def clear_conversation():
            """Clear chat and reset state."""
            tools.clear_reasoning()
            reasoning_capture.clear()
            return (
                [],  # chatbot
                "**📊 Status:** Ready to analyze",  # status
                None,  # metrics
                None,  # stats
                gr.update(visible=False),  # metrics visibility
                gr.update(visible=False),  # stats visibility
            )

        clear_btn.click(
            clear_conversation, 
            outputs=[chatbot, status_text, metrics_display, stats_display, metrics_display, stats_display]
        )

        # Main response handler
        def respond(
            message: str,
            history: List,
            loaded_state: bool,
            selected_data: Optional[str]
        ) -> Iterator:
            """
            Main chat handler with enhanced reasoning step display.
            """
            if not message.strip():
                return

            # Update status
            yield yield_state(history, loaded_state, "**📊 Status:** Loading data...")

            # Auto-load data if needed
            if not loaded_state or hermes_app.current_df is None or hermes_app.smart_df is None:
                try:
                    df, load_msg = hermes_app.load_data("Select Existing", None, selected_data)
                    
                    if df is None:
                        history = add_error_message(history, f"Unable to load data: {load_msg}")
                        yield yield_state(history, loaded_state, "**📊 Status:** ❌ Data loading failed")
                        return
                    
                    loaded_state = True
                    logger.info(f"✅ Data auto-loaded: {load_msg}")
                    
                except Exception as e:
                    logger.error(f"Auto-load error: {e}")
                    history = add_error_message(history, f"Error loading data: {str(e)}")
                    yield yield_state(history, loaded_state, "**📊 Status:** ❌ Error")
                    return

            # Add user message
            history.append(ChatMessage(role="user", content=message))
            yield yield_state(history, loaded_state, "**📊 Status:** 🔍 Classifying query...")

            # Clear previous logs
            reasoning_capture.clear()
            tools.clear_reasoning()

            # Classification
            try:
                classify_result = tools.classify_query_tool(message)
                detected_intent = classify_result.get("intent", "unknown")
                confidence = classify_result.get("confidence", 0)
                duration = classify_result.get("duration", 0)
                
                classification_content = f"""**Intent:** {detected_intent}  
**Confidence:** {confidence:.0%}  
**Duration:** {duration:.2f}s"""

                classification_content += f"\n✅ **Completed in {duration:.2f}s**"
                
                history = add_assistant_message(
                    history,
                    classification_content,
                    metadata={"title": "🎯 **Query Classification**"}
                )
                yield yield_state(history, loaded_state, f"**📊 Status:** 🧠 Processing {detected_intent} query...")

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Classification error: {e}")
                history = add_error_message(history, f"Classification failed: {str(e)}")
                yield yield_state(history, loaded_state, "**📊 Status:** ❌ Classification error")
                return

            # Query processing
            try:
                result = tools.analyze_query(message, detected_intent)
                
                if not result["success"]:
                    raise Exception(result["reasoning"])

                response_obj = result["response_model"]
                if response_obj is None:
                    raise Exception("No response object returned")

                # Display processing steps
                step_summary = reasoning_capture.get_step_summary()
                
                # Query Understanding
                if 'query_understanding' in reasoning_capture.categorized_steps:
                    query_logs = reasoning_capture.categorized_steps['query_understanding']
                    if query_logs:
                        duration_ms = query_logs[-1]['elapsed_ms'] - query_logs[0]['elapsed_ms']
                        duration_sec = duration_ms / 1000
                        
                        understanding_content = "**Analysis:**\n"
                        for log in query_logs[:3]:
                            msg = log.get('message', '').strip()
                            if msg and len(msg) < 200:
                                understanding_content += f"- {msg}\n"
                        
                        understanding_content += f"\n✅ **Completed in {duration_sec:.2f}s**"
                        
                        history = add_assistant_message(
                            history,
                            understanding_content,
                            metadata={"title": "🧠 **Query Understanding**"}
                        )
                        yield yield_state(history, loaded_state, "**📊 Status:** ⚙️ Generating code...")
                
                # Code Generation (with PROMPT restored)
                if 'code_generation' in reasoning_capture.categorized_steps:
                    code_logs = reasoning_capture.categorized_steps['code_generation']
                    if code_logs and len(code_logs) > 2:
                        duration_ms = code_logs[-1]['elapsed_ms'] - code_logs[0]['elapsed_ms']
                        duration_sec = duration_ms / 1000
                        
                        display_content = ""
                        
                        # Extract and show PROMPT (RESTORED)
                        if len(code_logs) > 1:
                            prompt = reasoning_capture.extract_code_from_message(
                                code_logs[1]["message"], "using prompt:"
                            )
                            if prompt:
                                # Limit prompt length for display
                                prompt_lines = prompt.strip().split('\n')
                                if len(prompt_lines) > 20:
                                    prompt = '\n'.join(prompt_lines[:20]) + '\n... (truncated)'
                                display_content += "**Prompt:**\n```\n" + prompt.strip() + "\n```\n\n"
                        
                        # Extract generated code
                        code = reasoning_capture.extract_code_from_message(
                            code_logs[2]["message"], "code generated:"
                        )
                        if code:
                            display_content += "**Generated Code:**\n```python\n" + code.strip() + "\n```\n\n"
                        
                        display_content += f"✅ **Generated in {duration_sec:.2f}s**"
                                
                        history = add_assistant_message(
                            history,
                            display_content,
                            metadata={"title": "⚙️ **Code Generation**"},
                        )
                        yield yield_state(history, loaded_state, "**📊 Status:** ✅ Validating code...")
                
                # Code Validation
                if 'code_validation' in reasoning_capture.categorized_steps:
                    validation_logs = reasoning_capture.categorized_steps['code_validation']
                    if validation_logs:
                        duration_ms = validation_logs[-1]['elapsed_ms'] - validation_logs[0]['elapsed_ms']
                        duration_sec = duration_ms / 1000
                        
                        validation_content = "**Checks:**\n"
                        for log in validation_logs[:3]:
                            msg = log.get('message', '').strip()
                            if msg and len(msg) < 150:
                                validation_content += f"- {msg}\n"
                        
                        validation_content += f"\n✅ **Validated in {duration_sec:.2f}s**"
                        
                        history = add_assistant_message(
                            history,
                            validation_content,
                            metadata={"title": "✅ **Code Validation**"}
                        )
                        yield yield_state(history, loaded_state, "**📊 Status:** 🚀 Executing code...")
                
                # Code Execution
                if 'code_execution' in reasoning_capture.categorized_steps:
                    exec_logs = reasoning_capture.categorized_steps['code_execution']
                    if exec_logs:
                        duration_ms = exec_logs[-1]['elapsed_ms'] - exec_logs[0]['elapsed_ms']
                        duration_sec = duration_ms / 1000
                        
                        exec_content = ""
                        
                        # Extract executing code
                        code = reasoning_capture.extract_code_from_message(
                            exec_logs[0]["message"], "executing code:"
                        )
                        if code:
                            exec_content += "**Executing Code:**\n```python\n" + code.strip() + "\n```\n\n"

                        try:
                            exec_content += "**Code Executed:**\n```python\n" + str(json.dumps(response_obj.raw_result.to_dict(), cls=CustomJsonEncoder, indent=2))[:100] + "\n```\n\n"
                        except Exception as e:
                            logger.warning(f"Failed to add executed code snippet: {e}")

                        exec_content += f"✅ **Executed in {duration_sec:.2f}s**"

                        history = add_assistant_message(
                            history,
                            exec_content,
                            metadata={"title": "🚀 **Code Execution**"}
                        )
                        yield yield_state(history, loaded_state, "**📊 Status:** 📊 Formatting results...")
                
                # Extract and display results
                components = extract_response_components(response_obj)
                
                # Text response
                if components["text"]:
                    history.append(ChatMessage(
                        role="assistant",
                        content=components["text"],
                    ))
                    yield yield_state(history, loaded_state, "**📊 Status:** ✅ Complete")
                
                # Number inline (FIXED - using HTML for better display)
                if components["data_type"] == "number" and components["number"] is not None:
                    try:
                        number_value = components["number"]
                        # Format number with commas
                        formatted_number = f"{number_value:,.2f}" if isinstance(number_value, float) else f"{number_value:,}"
                        
                        # Create styled HTML display
                        number_html = f"""
                        <div class="number-result">
                            🔢 {formatted_number}
                        </div>
                        """
                        
                        history.append(ChatMessage(
                            role="assistant",
                            content=number_html,
                        ))
                        logger.info(f"Added Number: {number_value}")
                        yield yield_state(history, loaded_state, "**📊 Status:** ✅ Complete")
                    except Exception as e:
                        logger.warning(f"Failed to add Number inline: {e}")

                # DataFrame inline
                if components["data_type"] == "dataframe" and components["dataframe"] is not None:
                    try:
                        df = components["dataframe"]
                        df_display = gr.DataFrame(
                            value=df,
                            label="📊 Results",
                            interactive=False,
                            wrap=True
                        )
                        
                        history.append(ChatMessage(
                            role="assistant",
                            content=df_display,
                        ))
                        logger.info(f"Added DataFrame with {len(df)} rows")
                        yield yield_state(history, loaded_state, "**📊 Status:** ✅ Complete")
                    except Exception as df_err:
                        logger.warning(f"Failed to add DataFrame inline: {df_err}")
                
                # Chart inline
                if components["chart_path"] and os.path.exists(components["chart_path"]):
                    try:
                        history.append(ChatMessage(
                            role="assistant",
                            content={"path": components["chart_path"], "mime_type": "image/png"},
                        ))
                        logger.info(f"Added chart: {components['chart_path']}")
                        yield yield_state(history, loaded_state, "**📊 Status:** ✅ Complete")
                    except Exception as chart_err:
                        logger.warning(f"Failed to add chart: {chart_err}")

                # Side panel data
                show_metrics = components["metrics"] is not None
                show_stats = components["stats"] is not None
                
                tools.clear_reasoning()

                yield yield_state(
                    history, 
                    loaded_state,
                    "**📊 Status:** ✅ Analysis complete",
                    metrics=components["metrics"] if show_metrics else None,
                    stats=components["stats"] if show_stats else None,
                    show_metrics=show_metrics,
                    show_stats=show_stats
                )

            except Exception as e:
                traceback.print_exc()
                logger.exception(f"Query processing error: {e}")
                history = add_error_message(history, f"Processing failed: {str(e)}")
                yield yield_state(history, loaded_state, "**📊 Status:** ❌ Error occurred")

        # Connect events
        outputs = [
            chatbot,
            status_text,
            metrics_display,
            stats_display,
            metrics_display,  # visibility
            stats_display,  # visibility
            data_loaded_state,
        ]

        # Submit handlers
        msg_input.submit(
            respond, 
            [msg_input, chatbot, data_loaded_state, data_selector], 
            outputs
        ).then(lambda: "", None, msg_input)
        
        submit_btn.click(
            respond, 
            [msg_input, chatbot, data_loaded_state, data_selector], 
            outputs
        ).then(lambda: "", None, msg_input)
        
        # Reload suggestions
        reload_btn.click(
            fn=reload_suggestions,
            inputs=[questions_state],
            outputs=suggestion_btns,
        )

        # Suggestion button handlers - use button's own value
        for btn in suggestion_btns:
            btn.click(
                fn=lambda btn_val: btn_val,
                inputs=[btn],
                outputs=msg_input,
            ).then(
                respond, 
                [msg_input, chatbot, data_loaded_state, data_selector], 
                outputs
            ).then(lambda: "", None, msg_input)
        
        # Initialize suggestions on load
        demo.load(
            fn=reload_suggestions,
            inputs=[questions_state],
            outputs=suggestion_btns,
        )

    return demo