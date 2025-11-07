"""
Gradio 5.48.0 Chat Interface with Intelligent Component Routing and Agent Tool Display

This UI automatically selects the best Gradio component for each response type:
- DataFrames → gr.DataFrame
- Numbers → gr.Number with highlighting
- Charts → gr.Image in chat
- Text → gr.Markdown in chat
- Agent reasoning → Displayed with metadata titles
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
import pandas as pd

# Import app types
from .models import (
    BaseResponse
)

from .utils import (
    load_questions_dataset,
    get_random_suggestions,
    extract_response_components,
    LLMReasoningCapture
)



# Initialize the reasoning capture handler
reasoning_capture = LLMReasoningCapture()

# Attach to relevant loggers
for logger_name in ['hermes', 'pandasai', 'litellm', 'root']:
    logging.getLogger(logger_name).addHandler(reasoning_capture)


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
    
    def clear_reasoning(self):
        """Clear reasoning history."""
        self.reasoning_steps = []


def create_agent_chat_interface(hermes_app) -> gr.Blocks:
    """
    Create Gradio interface with intelligent component routing and agent tool display.
    
    Different response types automatically display in optimal components:
    - Text → Chatbot
    - DataFrames → gr.DataFrame
    - Numbers → gr.Number  
    - Charts → gr.Image (inline in chat)
    - Metrics → gr.JSON
    - Agent reasoning → ChatMessage with metadata and step-by-step display
    """

    tools = HermesAgentTools(hermes_app)

    custom_css = """
    .chat-container {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background: white;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #0066cc 0%, #004080 100%);
        color: white;
        border-radius: 8px 8px 0 0;
        margin-bottom: 20px;
    }
    .number-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        margin: 10px 0;
    }
    .metrics-panel {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
        border: 1px solid #e0e0e0;
    }
    .dataframe-section {
        margin: 12px 0;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }
    /* Improved agent message styling */
    .message.bot.with-metadata {
        background: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 12px;
        margin: 8px 0;
    }
    .input-row {
        margin-top: 16px;
        gap: 8px;
    }
    /* Responsive layout */
    @media (max-width: 768px) {
        .header {
            padding: 12px;
        }
        .metrics-panel {
            padding: 12px;
        }
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

        # Hidden state for data loading
        data_loaded_state = gr.State(False)
        
        # Load questions dataset
        from .config import DATA_DIR
        all_questions = load_questions_dataset(DATA_DIR)
        questions_state = gr.State(all_questions)
        
        # Initialize with random suggestions
        initial_suggestions = get_random_suggestions(all_questions, 3)
        current_suggestions = gr.State(initial_suggestions)

        with gr.Row():
            with gr.Column(scale=7):
                # Main chatbot for text/images
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
                )
                
                # Number display (shown when number result)
                number_display = gr.Number(
                    label="🔢 Numeric Result",
                    visible=False,
                    interactive=False,
                    elem_classes="number-display",
                    # scale=2,
                )
                
                # DataFrame display (shown when DataFrame result)
                dataframe_display = gr.DataFrame(
                    label="📊 Data Table",
                    visible=False,
                    interactive=False,
                    wrap=True,
                    elem_classes="dataframe-section",
                )

            with gr.Column(scale=3):
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

        # Input area
        with gr.Row(elem_classes="input-row"):
            csv_files = hermes_app.get_csv_files()
            data_selector = gr.Dropdown(
                choices=[(f.name, str(f)) for f in csv_files],
                label="",
                value=str(csv_files[0]) if csv_files else None,
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

        # Dynamic suggestions from questions.csv
        gr.Markdown("**💡 Try asking:**")
        
        with gr.Row():
            suggestion_1 = gr.Button(
                "Loading...",
                scale=1,
                size="sm",
                variant="secondary",
                elem_id="suggestion_1",
            )
            suggestion_2 = gr.Button(
                "Loading...",
                scale=1,
                size="sm",
                variant="secondary",
                elem_id="suggestion_2",
            )
            suggestion_3 = gr.Button(
                "Loading...",
                scale=1,
                size="sm",
                variant="secondary",
                elem_id="suggestion_3",
            )
            reload_suggestions_btn = gr.Button(
                "🔄",
                scale=0.3,
                size="sm",
                variant="secondary",
                elem_id="reload_suggestions",
            )

        # Utility buttons
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", scale=1, size="sm", variant="secondary")

        # =====================================================================
        # Helper Functions
        # =====================================================================
        
        def create_empty_state():
            """Create empty state tuple for all outputs."""
            return (
                None,  # metrics
                None,  # stats
                None,  # number
                None,  # dataframe
                gr.update(visible=False),  # metrics visibility
                gr.update(visible=False),  # stats visibility
                gr.update(visible=False),  # number visibility
                gr.update(visible=False),  # dataframe visibility
            )
        
        def yield_state(history, loaded_state, metrics=None, stats=None, number=None, dataframe=None,
                       show_metrics=False, show_stats=False, show_number=False, show_dataframe=False):
            """Helper to yield consistent state tuple."""
            return (
                history,
                metrics,
                stats,
                number,
                dataframe,
                gr.update(visible=show_metrics),
                gr.update(visible=show_stats),
                gr.update(visible=show_number),
                gr.update(visible=show_dataframe),
                loaded_state
            )
        
        def add_assistant_message(history, content, metadata=None):
            """Helper to add assistant message to history."""
            history.append(ChatMessage(
                role="assistant",
                content=content,
                metadata=metadata
            ))
            return history
        
        def add_error_message(history, error_msg):
            """Helper to add error message to history."""
            return add_assistant_message(
                history,
                f"❌ **Error:** {error_msg}"
            )
        
        def extract_and_display_reasoning_steps(history):
            """Extract reasoning logs and display as processing steps."""
            all_logs = reasoning_capture.reasoning_logs
            
            if not all_logs:
                return history
            
            # Get step summary
            step_summary = reasoning_capture.get_step_summary()
            
            # Show each major step
            for step in step_summary.get('steps', []):
                step_name = step['name']
                duration = step['duration_ms']
                
                # Format step content
                step_content = f"**Duration:** {duration}ms"
                
                history = add_assistant_message(
                    history,
                    step_content,
                    metadata={"title": f"{step_name}"}
                )
            
            return history
        
        # =====================================================================
        # Event Handlers
        # =====================================================================
        
        def reload_suggestions(questions: List[str]) -> tuple:
            """Generate new random suggestions from questions dataset."""
            try:
                new_suggestions = get_random_suggestions(questions, 3)
                
                # Pad with empty if less than 3 questions
                while len(new_suggestions) < 3:
                    new_suggestions.append("No more questions available")
                
                return (
                    new_suggestions,
                    new_suggestions[0],
                    new_suggestions[1],
                    new_suggestions[2],
                )
            except Exception as e:
                logger.error(f"Error reloading suggestions: {e}")
                return (
                    ["Error loading questions"] * 3,
                    "Error",
                    "Error",
                    "Error",
                )
        
        def clear_conversation():
            tools.clear_reasoning()
            return (
                [],  # chatbot
                None,  # metrics
                None,  # stats
                None,  # number
                None,  # dataframe
                gr.update(visible=False),  # metrics visibility
                gr.update(visible=False),  # stats visibility
                gr.update(visible=False),  # number visibility
                gr.update(visible=False),  # dataframe visibility
            )

        clear_btn.click(
            clear_conversation, 
            outputs=[
                chatbot, 
                metrics_display, 
                stats_display,
                number_display,
                dataframe_display,
                metrics_display, 
                stats_display,
                number_display,
                dataframe_display,
            ]
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
            
            Shows dynamic processing steps:
            1. Auto-load data (if needed)
            2. User message
            3. Query Classification
            4. LLM Processing (with step-by-step reasoning)
            5. Code Generation & Execution
            6. Final result with smart component display
            """

            if not message.strip():
                return

            # Step 0: Auto-load data on first query if not already loaded
            if not loaded_state or hermes_app.current_df is None or hermes_app.smart_df is None:
                try:
                    df, load_msg = hermes_app.load_data("Select Existing", None, selected_data)
                    
                    if df is None:
                        history = add_error_message(history, f"Unable to load data: {load_msg}")
                        yield yield_state(history, loaded_state)
                        return
                    
                    loaded_state = True
                    logger.info(f"✅ Data auto-loaded: {load_msg}")
                    
                except Exception as e:
                    logger.error(f"Auto-load error: {e}")
                    history = add_error_message(history, f"Error loading data: {str(e)}")
                    yield yield_state(history, loaded_state)
                    return

            # Step 1: Add user message
            history.append(ChatMessage(role="user", content=message))
            yield yield_state(history, loaded_state)

            # Step 2: Classification
            try:
                # Clear previous reasoning logs
                reasoning_capture.clear()
                
                classify_result = tools.classify_query_tool(message)
                detected_intent = classify_result.get("intent", "unknown")
                confidence = classify_result.get("confidence", 0)
                
                # Add classification result
                classification_content = f"""**Intent:** {detected_intent}  
**Confidence:** {confidence:.0%}"""
                
                history = add_assistant_message(
                    history,
                    classification_content,
                    metadata={"title": "🎯 Query Classification"}
                )
                yield yield_state(history, loaded_state)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Classification error: {e}")
                history = add_error_message(history, f"Error during classification: {str(e)}")
                yield yield_state(history, loaded_state)
                return

            # Step 3: Query processing with enhanced reasoning display
            try:
                # Process the query
                result = tools.analyze_query(message, detected_intent)
                
                if not result["success"]:
                    raise Exception(result["reasoning"])

                response_obj = result["response_model"]
                
                if response_obj is None:
                    raise Exception("No response object returned")

                # Display processing steps dynamically
                step_summary = reasoning_capture.get_step_summary()
                
                # Show code generation step with formatted code
                if 'code_generation' in reasoning_capture.categorized_steps:
                    code_logs = reasoning_capture.categorized_steps['code_generation']
                    for log in code_logs:
                        if 'code generated:' in log['message'].lower():
                            code_start = log['message'].lower().find('code generated:') + len('code generated:')
                            code_content = log['message'][code_start:].strip()
                            
                            duration = log['elapsed_ms']
                            history = add_assistant_message(
                                history,
                                f"**Duration:** {duration}ms\n\n```python\n{code_content[:500]}\n```",
                                metadata={"title": "⚙️ Code Generation"}
                            )
                            yield yield_state(history, loaded_state)
                            break
                
                # Show code execution step
                if 'code_execution' in reasoning_capture.categorized_steps:
                    exec_logs = reasoning_capture.categorized_steps['code_execution']
                    if exec_logs:
                        duration = exec_logs[-1]['elapsed_ms'] - exec_logs[0]['elapsed_ms']
                        history = add_assistant_message(
                            history,
                            f"**Duration:** {duration}ms\n\nCode executed successfully ✓",
                            metadata={"title": "🚀 Code Execution"}
                        )
                        yield yield_state(history, loaded_state)
                
                # Extract components for smart display
                components = extract_response_components(response_obj)
                
                # Add text response
                if components["text"]:
                    history.append(ChatMessage(
                        role="assistant",
                        content=components["text"],
                    ))
                
                # Add number inline
                if components["data_type"] == "number" and components["number"] is not None:
                    try:
                        number = gr.Number(
                            interactive=True,
                            value=components["number"],
                            label="🔢 Numeric Result",
                            wrap=True,
                        )
                        history.append(ChatMessage(
                            role="assistant",
                            content=number,
                        ))
                        logger.info(f"Added gr.Number() as content of ChatMessage")
                    except Exception as e:
                        logger.warning(f"Failed to add gr.Number() inline: {e}")

                # Add DataFrame inline
                if components["data_type"] == "dataframe" and components["dataframe"] is not None:
                    try:
                        df = components["dataframe"]
                        df_display = gr.DataFrame(interactive=True, value=df, wrap=True)
                        
                        history.append(ChatMessage(
                            role="assistant",
                            content=df_display,
                        ))
                        logger.info(f"Added DataFrame inline with {len(df)} rows")
                    except Exception as df_err:
                        logger.warning(f"Failed to add DataFrame inline: {df_err}")
                
                # Add chart inline if present
                if components["chart_path"] and os.path.exists(components["chart_path"]):
                    try:
                        history.append(ChatMessage(
                            role="assistant",
                            content={"path": components["chart_path"], "mime_type": "image/png"},
                        ))
                        logger.info(f"Added chart inline: {components['chart_path']}")
                    except Exception as chart_err:
                        logger.warning(f"Failed to add chart inline: {chart_err}")

                # Determine component visibility for side panels
                # Most components are now shown inline, only show side panels for metrics/stats
                show_metrics = components["metrics"] is not None
                show_stats = components["stats"] is not None
                
                # Clear tools reasoning
                tools.clear_reasoning()

                # Yield final state with side panel data
                yield yield_state(
                    history, 
                    loaded_state,
                    metrics=components["metrics"] if show_metrics else None,
                    stats=components["stats"] if show_stats else None,
                    show_metrics=show_metrics,
                    show_stats=show_stats
                )

            except Exception as e:
                traceback.print_exc()
                logger.exception(f"Query processing error: {e}")
                history = add_error_message(history, f"Error processing query: {str(e)}")
                yield yield_state(history, loaded_state)

        # Connect events
        outputs = [
            chatbot,
            metrics_display,
            stats_display,
            number_display,
            dataframe_display,
            metrics_display,  # visibility
            stats_display,  # visibility
            number_display,  # visibility
            dataframe_display,  # visibility
            data_loaded_state,
        ]

        # Enter key to submit
        msg_input.submit(
            respond, [msg_input, chatbot, data_loaded_state, data_selector], outputs
        ).then(lambda: "", None, msg_input)
        
        # Submit button
        submit_btn.click(
            respond, [msg_input, chatbot, data_loaded_state, data_selector], outputs
        ).then(lambda: "", None, msg_input)
        
        # Reload suggestions button
        reload_suggestions_btn.click(
            fn=reload_suggestions,
            inputs=[questions_state],
            outputs=[current_suggestions, suggestion_1, suggestion_2, suggestion_3],
        )

        # Suggestion buttons
        def use_suggestion(suggestions: List[str], index: int) -> str:
            """Add suggestion to input."""
            return suggestions[index]
        
        for i, btn in enumerate([suggestion_1, suggestion_2, suggestion_3]):
            btn.click(
                fn=lambda s, idx=i: use_suggestion(s, idx),
                inputs=current_suggestions,
                outputs=msg_input,
            ).then(
                respond, [msg_input, chatbot, data_loaded_state, data_selector], outputs
            ).then(lambda: "", None, msg_input)
        
        # Initialize suggestions on load
        demo.load(
            fn=reload_suggestions,
            inputs=[questions_state],
            outputs=[current_suggestions, suggestion_1, suggestion_2, suggestion_3],
        )

    return demo