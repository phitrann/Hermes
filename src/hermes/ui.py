"""
Gradio UI glue code. Keeps UI concerns separate from logic.
"""
import os
import gradio as gr
from .app import HermesApp  # import from package-level app (see app.py)
from .config import SHIPMENTS_FILE, QUESTIONS_FILE

def create_gradio_app():
    app = HermesApp()

    custom_css = """
    .gradio-container { font-family: 'Inter', sans-serif; }
    .main-header { text-align: center; background: linear-gradient(120deg,#1f77b4,#ff7f0e); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.2em; font-weight:700; margin-bottom:10px; }
    """

    with gr.Blocks(css=custom_css, title="Hermes AI Logistics Assistant") as demo:
        gr.Markdown("<div class='main-header'>🚀 Hermes AI Logistics Assistant</div>")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Data Source")
                data_input_mode = gr.Radio(["Select Existing Data CSV","Upload New Data CSV"], value="Select Existing Data CSV")
                data_dropdown = gr.Dropdown(choices=app.get_csv_files(), value=SHIPMENTS_FILE if os.path.exists(SHIPMENTS_FILE) else None)
                data_uploader = gr.File(file_types=[".csv"], visible=False)
                def switch_mode(mode):
                    if mode == "Upload New Data CSV":
                        return gr.Dropdown(visible=False), gr.File(visible=True)
                    return gr.Dropdown(visible=True), gr.File(visible=False)
                data_input_mode.change(fn=switch_mode, inputs=data_input_mode, outputs=[data_dropdown, data_uploader])
                gr.Markdown("### 💡 Sample Questions")
                question_dropdown = gr.Dropdown(choices=app.get_questions(), allow_custom_value=True)
                gr.Markdown("### ⚡ Quick Actions")
                predict_btn = gr.Button("🔮 Get Predictions")
                recommend_btn = gr.Button("💡 Get Recommendations")
                export_btn = gr.Button("💾 Export Chat History")
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Hermes Anything")
                user_prompt = gr.Textbox(placeholder="e.g., Predict next week delays or Visualize warehouse performance", lines=3)
                with gr.Row():
                    submit_btn = gr.Button("🚀 Analyze")
                    clear_btn = gr.Button("🗑️ Clear")
                response_output = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("📈 Visualization"):
                        chart_output = gr.Image(height=420)
                    with gr.Tab("📊 Statistics"):
                        stats_output = gr.JSON()
                    with gr.Tab("🗃️ Data Preview"):
                        data_preview = gr.Dataframe()
                    with gr.Tab("💬 Chat History"):
                        history_output = gr.JSON()

        submit_btn.click(fn=app.process_query, inputs=[data_input_mode, data_uploader, data_dropdown, question_dropdown, user_prompt], outputs=[response_output, chart_output, stats_output, data_preview, history_output])
        predict_btn.click(fn=app.get_predictions, outputs=response_output)
        recommend_btn.click(fn=app.get_recommendations, outputs=response_output)
        export_btn.click(fn=app.export_chat_history, outputs=gr.File())
        clear_btn.click(fn=lambda: ("", None, None, None, []), outputs=[response_output, chart_output, stats_output, data_preview, history_output])

    return demo