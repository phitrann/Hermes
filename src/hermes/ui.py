"""
Gradio UI glue code. Keeps UI concerns separate from logic.
ChatGPT-like simple conversation interface with inline charts and recommendations.
"""
import os
import gradio as gr
from .app import HermesApp
from .config import SHIPMENTS_FILE

def create_gradio_app():
    app = HermesApp()

    custom_css = """
    .gradio-container { font-family: 'Inter', sans-serif; max-width: 900px; margin: auto; }
    .main-header { text-align: center; background: linear-gradient(120deg,#1f77b4,#ff7f0e); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.5em; font-weight:700; margin-bottom:20px; }
    .suggestion-btn { margin: 4px; }
    """

    with gr.Blocks(css=custom_css, title="Hermes AI Assistant") as demo:
        gr.Markdown("<div class='main-header'>🚀 Hermes AI Assistant</div>")
        gr.Markdown("<p style='text-align:center; color:#666; margin-bottom:20px;'>Your AI-powered logistics analytics assistant</p>")
        
        # Collapsible settings at the top
        with gr.Accordion("⚙️ Settings", open=False):
            with gr.Row():
                data_dropdown = gr.Dropdown(
                    choices=app.get_csv_files(), 
                    value=SHIPMENTS_FILE if os.path.exists(SHIPMENTS_FILE) else None,
                    label="Data Source",
                    interactive=True
                )
                clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
        
        # Main chat interface
        chatbot = gr.Chatbot(
            label="",
            height=500,
            show_label=False,
            avatar_images=(None, "🤖")
        )
        
        # Input area
        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Ask me anything about your shipments...",
                show_label=False,
                container=False,
                scale=9
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Suggested queries section - shown when chat is empty
        suggestions = gr.HTML(
            value="""
            <div style='margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px;'>
                <p style='margin-bottom: 10px; font-weight: 600; color: #333;'>💡 Try asking:</p>
                <ul style='list-style: none; padding: 0; margin: 0;'>
                    <li style='padding: 8px 0; color: #555;'>• Which route had the most delays last week?</li>
                    <li style='padding: 8px 0; color: #555;'>• Show warehouse performance metrics</li>
                    <li style='padding: 8px 0; color: #555;'>• Visualize delivery time trends</li>
                    <li style='padding: 8px 0; color: #555;'>• What's the average delay by route?</li>
                    <li style='padding: 8px 0; color: #555;'>• Compare warehouse processing times</li>
                </ul>
            </div>
            """,
            visible=True
        )
        
        # Quick action buttons
        with gr.Row(visible=True) as quick_actions:
            predict_btn = gr.Button("🔮 Get Predictions", size="sm")
            recommend_btn = gr.Button("💡 Get Recommendations", size="sm")
            stats_btn = gr.Button("📊 Show Statistics", size="sm")

        def handle_submit(message, data_source, history):
            if not message or not message.strip():
                return history, history, "", gr.update(visible=len(history) == 0), gr.update(visible=True)
            
            # Load data if needed
            if app.current_df is None:
                df, msg = app.load_data("Select Existing Data CSV", None, data_source)
                if df is None:
                    history.append([message, msg])
                    return history, history, "", gr.update(visible=False), gr.update(visible=True)
            
            # Process the query - returns (formatted_html, chart, stats, preview, chat_history)
            formatted_html, chart, stats, preview, chat_history = app.process_query(
                "Select Existing Data CSV", None, data_source, None, message
            )
            
            # Build the assistant's response with embedded chart if available
            response = formatted_html
            if chart is not None:
                # Chart will be embedded in the message - Gradio Chatbot supports tuples (text, image)
                history.append([message, (response, chart)])
            else:
                history.append([message, response])
            
            # Hide suggestions after first message
            return history, history, "", gr.update(visible=False), gr.update(visible=True)

        def handle_quick_action(action_type, data_source, history):
            if app.current_df is None:
                df, msg = app.load_data("Select Existing Data CSV", None, data_source)
                if df is None:
                    history.append(["Quick Action", msg])
                    return history, history, gr.update(visible=False), gr.update(visible=True)
            
            if action_type == "predictions":
                prompt = "Get predictions for next week"
                result = app.get_predictions()
            elif action_type == "recommendations":
                prompt = "Get recommendations"
                result = app.get_recommendations()
            else:  # statistics
                prompt = "Show me key statistics"
                if app.analytics:
                    stats = app.analytics.get_summary_stats()
                    result = f"""<strong>📊 Key Statistics</strong><br><br>
Total Shipments: {stats.get('total_shipments', 0)}<br>
Delayed Shipments: {stats.get('delayed_shipments', 0)}<br>
On-Time Rate: {stats.get('on_time_rate', 0):.1%}<br>
Average Delay: {stats.get('avg_delay_minutes', 0):.2f} minutes<br>
Median Delay: {stats.get('median_delay_minutes', 0):.2f} minutes"""
                else:
                    result = "Please load data first"
            
            history.append([prompt, result])
            return history, history, gr.update(visible=False), gr.update(visible=True)

        def handle_clear():
            app.chat_history = []
            return [], [], gr.update(visible=True), gr.update(visible=True)

        # Event handlers
        submit_btn.click(
            fn=handle_submit,
            inputs=[user_input, data_dropdown, chatbot],
            outputs=[chatbot, chatbot, user_input, suggestions, quick_actions]
        )
        
        user_input.submit(
            fn=handle_submit,
            inputs=[user_input, data_dropdown, chatbot],
            outputs=[chatbot, chatbot, user_input, suggestions, quick_actions]
        )
        
        predict_btn.click(
            fn=lambda ds, h: handle_quick_action("predictions", ds, h),
            inputs=[data_dropdown, chatbot],
            outputs=[chatbot, chatbot, suggestions, quick_actions]
        )
        
        recommend_btn.click(
            fn=lambda ds, h: handle_quick_action("recommendations", ds, h),
            inputs=[data_dropdown, chatbot],
            outputs=[chatbot, chatbot, suggestions, quick_actions]
        )
        
        stats_btn.click(
            fn=lambda ds, h: handle_quick_action("statistics", ds, h),
            inputs=[data_dropdown, chatbot],
            outputs=[chatbot, chatbot, suggestions, quick_actions]
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot, chatbot, suggestions, quick_actions]
        )

    return demo