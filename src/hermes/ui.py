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
    .gradio-container { font-family: 'Inter', sans-serif; max-width: 1400px; margin: auto; padding: 20px; }
    .main-header { text-align: center; background: linear-gradient(120deg,#1f77b4,#ff7f0e); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.5em; font-weight:700; margin-bottom:20px; }
    .suggestion-btn { margin: 4px; }
    @media (max-width: 768px) {
        .gradio-container { max-width: 100%; padding: 10px; }
        .main-header { font-size: 1.8em; }
    }
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
        with gr.Row(visible=True) as suggestions:
            gr.Markdown("### 💡 Try asking:")
        
        with gr.Row(visible=True) as suggestion_buttons:
            suggestion_1 = gr.Button("Which route had the most delays last week?", size="sm")
            suggestion_2 = gr.Button("Show warehouse performance metrics", size="sm")
        
        with gr.Row(visible=True) as suggestion_buttons_2:
            suggestion_3 = gr.Button("Visualize delivery time trends", size="sm")
            suggestion_4 = gr.Button("What's the average delay by route?", size="sm")
        
        with gr.Row(visible=True) as suggestion_buttons_3:
            suggestion_5 = gr.Button("Compare warehouse processing times", size="sm")
        
        # Quick action buttons
        with gr.Row(visible=True) as quick_actions:
            predict_btn = gr.Button("🔮 Get Predictions", size="sm")
            recommend_btn = gr.Button("💡 Get Recommendations", size="sm")
            stats_btn = gr.Button("📊 Show Statistics", size="sm")

        def handle_submit(message, data_source, history):
            if not message or not message.strip():
                return history, "", gr.update(visible=len(history) == 0), gr.update(visible=len(history) == 0), gr.update(visible=len(history) == 0), gr.update(visible=len(history) == 0), gr.update(visible=True)
            
            # Load data if needed
            if app.current_df is None:
                df, msg = app.load_data("Select Existing Data CSV", None, data_source)
                if df is None:
                    history.append([message, msg])
                    return history, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            
            # Process the query - returns (formatted_html, chart, stats, preview, chat_history)
            formatted_html, chart, stats, preview, chat_history = app.process_query(
                "Select Existing Data CSV", None, data_source, None, message
            )
            
            # Build the assistant's response with embedded chart if available
            response = formatted_html
            if chart is not None:
                # Chart will be embedded in the message
                # Gradio Chatbot supports tuples (text, image) for inline images (requires Gradio >= 3.x)
                # chart is a PIL Image object returned from app.process_query
                history.append([message, (response, chart)])
            else:
                history.append([message, response])
            
            # Hide suggestions after first message
            return history, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        def handle_quick_action(action_type, data_source, history):
            if app.current_df is None:
                df, msg = app.load_data("Select Existing Data CSV", None, data_source)
                if df is None:
                    history.append(["Quick Action", msg])
                    return history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            
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
On-Time Rate: {stats.get('on_time_rate', 0) * 100:.1f}%<br>
Average Delay: {stats.get('avg_delay_minutes', 0):.2f} minutes<br>
Median Delay: {stats.get('median_delay_minutes', 0):.2f} minutes"""
                else:
                    result = "Please load data first"
            
            history.append([prompt, result])
            return history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        def handle_clear():
            app.chat_history = []
            return [], gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

        # Event handlers
        submit_btn.click(
            fn=handle_submit,
            inputs=[user_input, data_dropdown, chatbot],
            outputs=[chatbot, user_input, suggestions, suggestion_buttons, suggestion_buttons_2, suggestion_buttons_3, quick_actions]
        )
        
        user_input.submit(
            fn=handle_submit,
            inputs=[user_input, data_dropdown, chatbot],
            outputs=[chatbot, user_input, suggestions, suggestion_buttons, suggestion_buttons_2, suggestion_buttons_3, quick_actions]
        )
        
        # Suggestion button handlers - helper function to reduce duplication
        def create_suggestion_handler(query):
            return lambda ds, h: handle_submit(query, ds, h)
        
        suggestion_outputs = [chatbot, user_input, suggestions, suggestion_buttons, suggestion_buttons_2, suggestion_buttons_3, quick_actions]
        
        suggestion_1.click(
            fn=create_suggestion_handler("Which route had the most delays last week?"),
            inputs=[data_dropdown, chatbot],
            outputs=suggestion_outputs
        )
        
        suggestion_2.click(
            fn=create_suggestion_handler("Show warehouse performance metrics"),
            inputs=[data_dropdown, chatbot],
            outputs=suggestion_outputs
        )
        
        suggestion_3.click(
            fn=create_suggestion_handler("Visualize delivery time trends"),
            inputs=[data_dropdown, chatbot],
            outputs=suggestion_outputs
        )
        
        suggestion_4.click(
            fn=create_suggestion_handler("What's the average delay by route?"),
            inputs=[data_dropdown, chatbot],
            outputs=suggestion_outputs
        )
        
        suggestion_5.click(
            fn=create_suggestion_handler("Compare warehouse processing times"),
            inputs=[data_dropdown, chatbot],
            outputs=suggestion_outputs
        )
        
        predict_btn.click(
            fn=lambda ds, h: handle_quick_action("predictions", ds, h),
            inputs=[data_dropdown, chatbot],
            outputs=[chatbot, suggestions, suggestion_buttons, suggestion_buttons_2, suggestion_buttons_3, quick_actions]
        )
        
        recommend_btn.click(
            fn=lambda ds, h: handle_quick_action("recommendations", ds, h),
            inputs=[data_dropdown, chatbot],
            outputs=[chatbot, suggestions, suggestion_buttons, suggestion_buttons_2, suggestion_buttons_3, quick_actions]
        )
        
        stats_btn.click(
            fn=lambda ds, h: handle_quick_action("statistics", ds, h),
            inputs=[data_dropdown, chatbot],
            outputs=[chatbot, suggestions, suggestion_buttons, suggestion_buttons_2, suggestion_buttons_3, quick_actions]
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot, suggestions, suggestion_buttons, suggestion_buttons_2, suggestion_buttons_3, quick_actions]
        )

    return demo