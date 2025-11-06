"""
Gradio UI glue code. Keeps UI concerns separate from logic.
Conversational chatbot interface similar to Claude/ChatGPT/Grok.
"""
import os
import shutil
import tempfile
import time
import traceback
import gradio as gr
from .app import HermesApp
from .config import SHIPMENTS_FILE

def create_gradio_app():
    app = HermesApp()

    custom_css = """
    .gradio-container { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .main-header { 
        text-align: center; 
        color: #667eea;
        font-size: 2.2em; 
        font-weight: 900; 
        margin: 20px 0 5px 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95em;
        margin-bottom: 20px;
    }
    .chat-section {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .control-panel {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .status-badge {
        padding: 10px 16px;
        border-radius: 8px;
        background: #f0f4f8;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        font-size: 0.9em;
    }
    .suggestion-btn {
        font-size: 0.85em !important;
        padding: 8px 14px !important;
        border-radius: 20px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    .action-btn {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button {
        transition: all 0.2s ease !important;
    }
    textarea {
        border-radius: 10px !important;
        border: 2px solid #e5e7eb !important;
    }
    textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    .tab-content {
        padding: 20px !important;
    }
    """

    with gr.Blocks(css=custom_css, title="Hermes AI - Logistics Analytics Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("<div class='main-header'>� Hermes AI</div>")
        gr.Markdown("<div class='subtitle'>Ask a question in natural language to get SQL + results.</div>")
        
        # Hidden state to track data loading status
        data_loaded_state = gr.State(False)
        
        with gr.Row():
            # Sidebar with data controls and quick actions
            with gr.Column(scale=1, min_width=320):
                with gr.Group(elem_classes="sidebar-section"):
                    gr.Markdown("### 📁 Data Source")
                    data_input_mode = gr.Radio(
                        ["Select Existing", "Upload New"], 
                        value="Select Existing",
                        label="Mode",
                        container=False
                    )
                    data_dropdown = gr.Dropdown(
                        choices=app.get_csv_files(), 
                        value=SHIPMENTS_FILE if os.path.exists(SHIPMENTS_FILE) else None,
                        label="Select CSV",
                        container=False
                    )
                    data_uploader = gr.File(
                        file_types=[".csv"], 
                        label="Upload CSV",
                        visible=False,
                        container=False
                    )
                    load_data_btn = gr.Button("📥 Load Data", variant="primary", elem_classes="action-btn")
                    data_status = gr.Markdown("*No data loaded*")
                
                with gr.Group(elem_classes="sidebar-section"):
                    gr.Markdown("### ⚡ Quick Actions")
                    predict_btn = gr.Button("🔮 Predict Delays", elem_classes="action-btn", size="sm")
                    recommend_btn = gr.Button("💡 Get Recommendations", elem_classes="action-btn", size="sm")
                    stats_btn = gr.Button("📊 Show Statistics", elem_classes="action-btn", size="sm")
                
                with gr.Group(elem_classes="sidebar-section"):
                    gr.Markdown("### 💡 Sample Questions")
                    question_dropdown = gr.Dropdown(
                        choices=app.get_questions()[:10],  # Show first 10
                        label="Quick Questions",
                        container=False,
                        allow_custom_value=True
                    )
                    use_question_btn = gr.Button("Use This Question", size="sm")
                
                with gr.Group(elem_classes="sidebar-section"):
                    gr.Markdown("### �️ Tools")
                    export_btn = gr.Button("� Export History", size="sm")
                    clear_history_btn = gr.Button("�️ Clear Chat", size="sm")
            
            # Main chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    label="",
                    height=550,
                    show_label=False,
                    type='messages',  # Use modern messages format
                    avatar_images=(
                        "https://api.dicebear.com/7.x/avataaars/svg?seed=user",
                        "https://api.dicebear.com/7.x/bottts/svg?seed=hermes"
                    ),
                    bubble_full_width=False,
                    elem_classes="chat-container"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask me anything about your logistics data... (e.g., 'Show me delayed shipments', 'Predict next week trends')",
                        show_label=False,
                        scale=9,
                        container=False,
                        lines=1,
                        max_lines=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes="action-btn")
                
                with gr.Row():
                    gr.Markdown("**💡 Suggestions:**")
                with gr.Row():
                    suggestion_1 = gr.Button("📈 Visualize shipment trends", size="sm", elem_classes="suggestion-btn")
                    suggestion_2 = gr.Button("⏱️ Which routes have most delays?", size="sm", elem_classes="suggestion-btn")
                    suggestion_3 = gr.Button("🏢 Compare warehouse performance", size="sm", elem_classes="suggestion-btn")
        
        # Event handlers
        def switch_data_mode(mode):
            if mode == "Upload New":
                return gr.Dropdown(visible=False), gr.File(visible=True)
            return gr.Dropdown(visible=True), gr.File(visible=False)
        
        def load_data_handler(mode, uploaded, selected):
            """Load data and return status message"""
            df, msg = app.load_data(mode, uploaded, selected)
            if df is not None:
                return True, f"✅ {msg}"
            return False, f"❌ {msg}"
        
        def user_message_handler(message, history):
            """Add user message to chat"""
            if not message.strip():
                return "", history
            # Use messages format: list of dicts with 'role' and 'content'
            return "", history + [{"role": "user", "content": message}]
        
        def bot_response_handler(history, mode, uploaded, selected, loaded_state):
            """Generate bot response with typing indicator"""
            if not history:
                return history, loaded_state
            
            # Check if last message needs a response
            last_msg = history[-1]
            if last_msg.get("role") != "user":
                return history, loaded_state
            
            user_msg = last_msg.get("content", "")
            
            # Add temporary assistant message with typing indicator
            history.append({"role": "assistant", "content": "🤖 *Analyzing your request...*"})
            yield history, loaded_state
            time.sleep(0.3)
            
            # Ensure data is loaded
            if not loaded_state:
                history[-1]["content"] = "📥 *Loading data first...*"
                yield history, loaded_state
                
                df, status_msg = app.load_data(mode, uploaded, selected)
                if df is None:
                    history[-1]["content"] = f"❌ **Data Loading Failed**\n\n{status_msg}\n\nPlease use the 'Load Data' button in the sidebar to select or upload a CSV file."
                    yield history, False
                    return
                loaded_state = True
                history[-1]["content"] = "✅ *Data loaded! Processing your query...*"
                yield history, loaded_state
                time.sleep(0.3)
            
            # Process query
            history[-1]["content"] = "🔍 *Analyzing data and generating response...*"
            yield history, loaded_state
            
            try:
                response_dict = app.process_query_chat(user_msg)
                
                # Build response message with chart if present
                response_text = response_dict.get("text", "")
                chart_path = response_dict.get("chart", None)
                
                if chart_path and os.path.exists(chart_path):
                    # Copy to temp file to avoid Gradio path validation issues
                    temp_dir = tempfile.mkdtemp()
                    temp_filename = f"chart_{os.path.basename(chart_path)}"
                    temp_path = os.path.join(temp_dir, temp_filename)
                    shutil.copy2(chart_path, temp_path)
                    
                    # Return as multimodal message with text and image file
                    history[-1]["content"] = {
                        "text": response_text,
                        "path": temp_path, 
                        "mime_type": "image/png",
                    }
                else:
                    history[-1]["content"] = response_text
                
                yield history, loaded_state
            except Exception as e:
                error_detail = traceback.format_exc()
                history[-1]["content"] = f"❌ **Error**\n\nSomething went wrong: {str(e)}\n\nPlease try again or rephrase your question."
                yield history, loaded_state
        
        def use_sample_question(question, history):
            """Add sample question to chat"""
            if question:
                return "", history + [{"role": "user", "content": question}]
            return "", history
        
        def clear_chat():
            """Clear chat history"""
            app.chat_history = []
            return [], False, "*No data loaded*"
        
        def export_history_handler():
            """Export chat history"""
            return app.export_chat_history()
        
        def quick_action_handler(action_type, history):
            """Handle quick action buttons"""
            prompts = {
                "predict": "Predict delays for next week",
                "recommend": "Give me recommendations to improve logistics performance",
                "stats": "Show me summary statistics of the shipment data"
            }
            prompt = prompts.get(action_type, "")
            if prompt:
                return "", history + [{"role": "user", "content": prompt}]
            return "", history
        
        # Wire up events
        data_input_mode.change(
            fn=switch_data_mode, 
            inputs=data_input_mode, 
            outputs=[data_dropdown, data_uploader]
        )
        
        load_data_btn.click(
            fn=load_data_handler,
            inputs=[data_input_mode, data_uploader, data_dropdown],
            outputs=[data_loaded_state, data_status]
        )
        
        # Chat interaction - two-step process
        msg_submit = msg.submit(
            fn=user_message_handler,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False
        ).then(
            fn=bot_response_handler,
            inputs=[chatbot, data_input_mode, data_uploader, data_dropdown, data_loaded_state],
            outputs=[chatbot, data_loaded_state]
        )
        
        send_btn.click(
            fn=user_message_handler,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False
        ).then(
            fn=bot_response_handler,
            inputs=[chatbot, data_input_mode, data_uploader, data_dropdown, data_loaded_state],
            outputs=[chatbot, data_loaded_state]
        )
        
        # Sample question usage
        use_question_btn.click(
            fn=use_sample_question,
            inputs=[question_dropdown, chatbot],
            outputs=[msg, chatbot]
        )
        
        # Quick actions
        predict_btn.click(
            fn=lambda h: quick_action_handler("predict", h),
            inputs=[chatbot],
            outputs=[msg, chatbot]
        ).then(
            fn=bot_response_handler,
            inputs=[chatbot, data_input_mode, data_uploader, data_dropdown, data_loaded_state],
            outputs=[chatbot, data_loaded_state]
        )
        
        recommend_btn.click(
            fn=lambda h: quick_action_handler("recommend", h),
            inputs=[chatbot],
            outputs=[msg, chatbot]
        ).then(
            fn=bot_response_handler,
            inputs=[chatbot, data_input_mode, data_uploader, data_dropdown, data_loaded_state],
            outputs=[chatbot, data_loaded_state]
        )
        
        stats_btn.click(
            fn=lambda h: quick_action_handler("stats", h),
            inputs=[chatbot],
            outputs=[msg, chatbot]
        ).then(
            fn=bot_response_handler,
            inputs=[chatbot, data_input_mode, data_uploader, data_dropdown, data_loaded_state],
            outputs=[chatbot, data_loaded_state]
        )
        
        # Suggestion buttons
        def handle_suggestion(btn_text, history):
            return "", history + [{"role": "user", "content": btn_text}]
        
        for btn in [suggestion_1, suggestion_2, suggestion_3]:
            btn.click(
                fn=lambda h, b=btn: handle_suggestion(b.value, h),
                inputs=[chatbot],
                outputs=[msg, chatbot]
            ).then(
                fn=bot_response_handler,
                inputs=[chatbot, data_input_mode, data_uploader, data_dropdown, data_loaded_state],
                outputs=[chatbot, data_loaded_state]
            )
        
        # Utility actions
        clear_history_btn.click(
            fn=clear_chat,
            outputs=[chatbot, data_loaded_state, data_status]
        )
        
        export_btn.click(
            fn=export_history_handler,
            outputs=gr.File()
        )
    
    return demo