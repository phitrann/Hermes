"""
Hermes AI Logistics Assistant - Enhanced PandasAI Implementation
Built on PandasAI with advanced features for logistics analytics
"""

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import json
import pandas as pd 
import numpy as np
import logging
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PandasAI imports
import pandasai as pai
from pandasai import SmartDataframe
from pandasai.helpers.logger import Logger
from pandasai_litellm.litellm import LiteLLM

# ML imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# UI
import gradio as gr

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "./data"
LOGS_DIR = "./logs"
CACHE_DIR = "./.cache"
CHARTS_DIR = "./charts"
SHIPMENTS_FILE = os.path.join(DATA_DIR, "shipments.csv")
QUESTIONS_FILE = os.path.join(DATA_DIR, "shipment_questions_500.csv")

# Create necessary directories
for directory in [LOGS_DIR, CACHE_DIR, CHARTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'hermes.log')),
        logging.StreamHandler()
    ]
)
logger_app = logging.getLogger(__name__)

# PandasAI Configuration
pandasai_logger = Logger(save_logs=True, verbose=False)
llm = LiteLLM(
    model="openai/Qwen/Qwen3-14B-AWQ",
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
    temperature=0.1,
    max_tokens=2000
)

pai.config.set({
    "llm": llm,
    "logger": pandasai_logger,
    "enable_cache": True,
    "save_charts": True,
    "save_charts_path": CHARTS_DIR,
    "verbose": False,
    "enforce_privacy": True,
    "max_retries": 3
})

# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class HermesAnalytics:
    """Enhanced analytics engine with ML capabilities"""
    
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.prediction_model = None
        self.encoders = {}
        
    def get_summary_stats(self):
        """Get comprehensive summary statistics"""
        total = len(self.df)
        delayed = (self.df['delay_minutes'] > 0).sum()
        
        stats = {
            'total_shipments': total,
            'delayed_shipments': int(delayed),
            'on_time_shipments': int(total - delayed),
            'delay_rate': f"{(delayed / total * 100):.1f}%",
            'avg_delay': f"{self.df[self.df['delay_minutes'] > 0]['delay_minutes'].mean():.1f} min",
            'avg_delivery_time': f"{self.df['delivery_time'].mean():.2f} days",
            'date_range': f"{self.df['date'].min().date()} to {self.df['date'].max().date()}"
        }
        
        # Only include cost if column exists (fixes the error)
        if 'cost' in self.df.columns:
            stats['total_cost'] = f"${self.df['cost'].sum():,.2f}"
            
        return stats
    
    def train_prediction_model(self):
        """Train ML model for delay prediction"""
        try:
            df_model = self.df.copy()
            
            # Feature engineering
            df_model['day_of_week'] = df_model['date'].dt.dayofweek
            df_model['week_of_year'] = df_model['date'].dt.isocalendar().week
            df_model['month'] = df_model['date'].dt.month
            
            # Encode categoricals
            le_route = LabelEncoder()
            le_warehouse = LabelEncoder()
            
            df_model['route_encoded'] = le_route.fit_transform(df_model['route'])
            df_model['warehouse_encoded'] = le_warehouse.fit_transform(df_model['warehouse'])
            
            self.encoders = {'route': le_route, 'warehouse': le_warehouse}
            
            # Prepare features
            features = ['route_encoded', 'warehouse_encoded', 'day_of_week', 'week_of_year', 'month']
            X = df_model[features]
            y = df_model['delay_minutes']
            
            # Train model
            self.prediction_model = LinearRegression()
            self.prediction_model.fit(X, y)
            
            # Evaluate
            predictions = self.prediction_model.predict(X)
            r2 = r2_score(y, predictions)
            rmse = np.sqrt(mean_squared_error(y, predictions))
            
            logger_app.info(f"Model trained: R²={r2:.4f}, RMSE={rmse:.2f}")
            
            return {
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': dict(zip(features, self.prediction_model.coef_))
            }
        except Exception as e:
            logger_app.error(f"Model training failed: {e}")
            return None
    
    def predict_next_week(self):
        """Predict next week's average delay"""
        if self.prediction_model is None:
            self.train_prediction_model()
        
        if self.prediction_model is None:
            return None
        
        # Generate next week scenarios
        next_week_start = self.df['date'].max() + timedelta(days=1)
        predictions = []
        
        for day_offset in range(7):
            date = next_week_start + timedelta(days=day_offset)
            for route in self.df['route'].unique():
                for warehouse in self.df['warehouse'].unique():
                    features = np.array([[
                        self.encoders['route'].transform([route])[0],
                        self.encoders['warehouse'].transform([warehouse])[0],
                        date.dayofweek,
                        date.isocalendar()[1],
                        date.month
                    ]])
                    pred = self.prediction_model.predict(features)[0]
                    predictions.append(max(0, pred))
        
        return {
            'predicted_avg_delay': np.mean(predictions),
            'predicted_median': np.median(predictions),
            'confidence_interval': (np.percentile(predictions, 25), np.percentile(predictions, 75)),
            'forecast_period': f"{next_week_start.date()} to {(next_week_start + timedelta(days=6)).date()}"
        }
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Warehouse performance
        wh_perf = self.df.groupby('warehouse').agg({
            'delivery_time': 'mean',
            'delay_minutes': 'mean',
            'on_time': 'mean'
        }).reset_index()
        
        best_wh = wh_perf['delivery_time'].idxmin()
        worst_wh = wh_perf['delivery_time'].idxmax()
        
        recommendations.append({
            'category': 'Best Practice',
            'priority': 'High',
            'finding': f"{best_wh} has best delivery time ({wh_perf.loc[best_wh, 'delivery_time']:.2f} days)",
            'action': f"Document and replicate {best_wh}'s processes across other warehouses"
        })
        
        recommendations.append({
            'category': 'Needs Improvement',
            'priority': 'High',
            'finding': f"{worst_wh} has slowest delivery time ({wh_perf.loc[worst_wh, 'delivery_time']:.2f} days)",
            'action': f"Conduct process audit at {worst_wh} and implement efficiency improvements"
        })
        
        # Delay reasons
        delay_impact = self.df[self.df['delay_minutes'] > 0].groupby('delay_reason').agg({
            'id': 'count',
            'delay_minutes': 'mean',
            'cost': 'sum'
        }).sort_values('cost', ascending=False)
        
        top_delay = delay_impact.index[0]
        recommendations.append({
            'category': 'Cost Reduction',
            'priority': 'Critical',
            'finding': f"{top_delay} causes ${delay_impact.loc[top_delay, 'cost']:,.2f} in delay costs",
            'action': self._get_mitigation_strategy(top_delay)
        })
        
        # Route optimization
        route_perf = self.df.groupby('route')['on_time'].mean().sort_values()
        worst_route = route_perf.index[0]
        
        recommendations.append({
            'category': 'Route Optimization',
            'priority': 'Medium',
            'finding': f"{worst_route} has only {route_perf[worst_route]*100:.1f}% on-time delivery",
            'action': f"Review {worst_route} for scheduling, capacity, and infrastructure issues"
        })
        
        return recommendations
    
    def _get_mitigation_strategy(self, delay_reason):
        """Get mitigation strategy for delay reason"""
        strategies = {
            'Weather': 'Implement weather monitoring system, flexible scheduling, and alternative routing protocols',
            'Traffic': 'Deploy real-time traffic integration, optimize departure times, use traffic prediction AI',
            'Mechanical': 'Increase preventive maintenance frequency by 30%, implement predictive maintenance IoT',
            'Staff Shortage': 'Build backup staff pool, improve shift planning, implement cross-training program'
        }
        return strategies.get(delay_reason, 'Conduct root cause analysis and implement targeted interventions')

# ============================================================================
# VISUALIZATION ENGINE (Using PandasAI Charts)
# ============================================================================

class HermesVisualizer:
    """Visualization engine leveraging PandasAI chart features"""
    
    def __init__(self, charts_dir=CHARTS_DIR):
        self.charts_dir = charts_dir
    
    def get_latest_chart(self):
        """Get the most recently generated chart from PandasAI"""
        try:
            chart_files = glob.glob(os.path.join(self.charts_dir, "*.png"))
            if not chart_files:
                return None
            
            # Get the most recent chart file
            latest_chart = max(chart_files, key=os.path.getmtime)
            return latest_chart
        except Exception as e:
            logger_app.error(f"Error getting latest chart: {e}")
            return None

# ============================================================================
# GRADIO APPLICATION
# ============================================================================

class HermesApp:
    """Main application class"""
    
    def __init__(self):
        self.chat_history = []
        self.current_df = None
        self.smart_df = None
        self.analytics = None
        self.visualizer = HermesVisualizer()
        
    def get_csv_files(self):
        """Get list of CSV files in data directory"""
        return glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    def get_questions(self):
        """Load sample questions"""
        try:
            return pd.read_csv(QUESTIONS_FILE)["question"].tolist()
        except:
            return []
    
    def load_data(self, data_source_type, uploaded_file, selected_file):
        """Load data from file or upload"""
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
            df['date'] = pd.to_datetime(df['date'])
            
            self.current_df = df
            self.smart_df = SmartDataframe(df, config={"llm": llm, "enable_cache": True})
            self.analytics = HermesAnalytics(df)
            
            logger_app.info(f"Loaded {len(df)} records from {data_path}")
            return df, f"✅ Loaded {len(df)} shipment records"
        
        except Exception as e:
            logger_app.error(f"Data loading error: {e}")
            return None, f"❌ Error loading data: {str(e)}"
    
    def process_query(self, data_source_type, uploaded_file, selected_file, 
                     selected_question, user_prompt):
        """Main query processing function"""
        
        # Load data if not loaded
        if self.current_df is None or self.smart_df is None:
            df, msg = self.load_data(data_source_type, uploaded_file, selected_file)
            if df is None:
                return msg, None, None, None, self.chat_history
        
        # Construct full prompt
        prompt = selected_question if selected_question else ""
        if user_prompt:
            prompt = user_prompt if not prompt else f"{prompt}\n{user_prompt}"
        
        if not prompt:
            return "❌ Please enter a question or select one from the list", None, None, None, self.chat_history
        
        try:
            # Add time context
            max_date = self.current_df['date'].max().strftime('%Y-%m-%d')
            time_context = f"<TIME_CONTEXT>Current date in dataset: {max_date}</TIME_CONTEXT>\n"
            full_prompt = time_context + prompt + "/no_think"
            
            logger_app.info(f"Processing query: {prompt}")
            
            # Query PandasAI (which may generate charts automatically)
            response = self.smart_df.chat(full_prompt)
            
            # Get the latest chart generated by PandasAI (if any)
            chart_path = self.visualizer.get_latest_chart()
            
            # Get analytics
            stats = self.analytics.get_summary_stats()
            
            # Update chat history
            self.chat_history.append({
                'query': prompt,
                'response': str(response),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chart_generated': chart_path is not None
            })
            
            # Beautiful formatted response with HTML styling
            response_str = str(response).replace('\n', '<br>')
            formatted_response = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 2em; margin-right: 10px;">🤖</span>
        <div>
            <h3 style="margin: 0; color: white; font-weight: bold;">Hermes AI Response</h3>
            <small style="color: rgba(255,255,255,0.8);">Intelligent Logistics Assistant</small>
        </div>
    </div>
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px);">
        {response_str}
    </div>
    <div style="text-align: right; margin-top: 10px; font-size: 0.8em; color: rgba(255,255,255,0.7);">
        📊 Powered by PandasAI & Local LLM
    </div>
</div>
"""
            
            return formatted_response, chart_path, stats, self.current_df.head(10), self.chat_history
        
        except Exception as e:
            logger_app.error(f"Query processing error: {e}")
            error_msg = f"""
<div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 20px; border-radius: 15px; margin: 10px 0;">
    <span style="font-size: 2em;">❌</span>
    <strong>Error Processing Query</strong><br>
    {str(e)}
</div>
"""
            return error_msg, None, None, None, self.chat_history
    
    def get_predictions(self):
        """Get ML predictions"""
        if self.analytics is None:
            return "❌ Please load data first"
        
        try:
            # Train model
            model_metrics = self.analytics.train_prediction_model()
            
            # Get prediction
            prediction = self.analytics.predict_next_week()
            
            if prediction:
                result = f"""
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 15px; margin: 10px 0;">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 2em; margin-right: 10px;">🔮</span>
        <h3 style="margin: 0; color: white;">Prediction Results</h3>
    </div>
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
        <strong>📊 Model Performance:</strong><br>
        • R² Score: {model_metrics['r2_score']:.4f}<br>
        • RMSE: {model_metrics['rmse']:.2f} minutes<br><br>
        
        <strong>🔮 Next Week Forecast:</strong><br>
        • Predicted Average Delay: {prediction['predicted_avg_delay']:.2f} minutes<br>
        • Predicted Median: {prediction['predicted_median']:.2f} minutes<br>
        • Confidence Interval: ({prediction['confidence_interval'][0]:.2f}, {prediction['confidence_interval'][1]:.2f}) minutes<br>
        • Forecast Period: {prediction['forecast_period']}<br><br>
        
        <em>Note: Prediction based on historical patterns from {len(self.current_df)} shipments</em>
    </div>
</div>
"""
                return result
            else:
                return """
<div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 20px; border-radius: 15px; margin: 10px 0;">
    <span style="font-size: 2em;">❌</span>
    <strong>Prediction Failed</strong>
</div>
"""
        
        except Exception as e:
            logger_app.error(f"Prediction error: {e}")
            return f"""
<div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 20px; border-radius: 15px; margin: 10px 0;">
    <span style="font-size: 2em;">❌</span>
    <strong>Prediction Error:</strong> {str(e)}
</div>
"""
    
    def get_recommendations(self):
        """Get actionable recommendations"""
        if self.analytics is None:
            return "❌ Please load data first"
        
        try:
            recommendations = self.analytics.generate_recommendations()
            
            result = """
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 15px; margin: 10px 0;">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 2em; margin-right: 10px;">💡</span>
        <h3 style="margin: 0; color: white;">Hermes AI Recommendations</h3>
    </div>
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
"""
            
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
                emoji = priority_emoji.get(rec['priority'], '⚪')
                
                result += f"""
        <div style="background: rgba(255,255,255,0.1); padding: 10px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #fff;">
            <strong>{i}. {rec['category']}</strong> {emoji} <em>{rec['priority']} Priority</em><br><br>
            <strong>Finding:</strong> {rec['finding']}<br>
            <strong>Action:</strong> {rec['action']}
        </div>
"""
            
            result += """
    </div>
</div>
"""
            
            return result
        
        except Exception as e:
            logger_app.error(f"Recommendations error: {e}")
            return f"""
<div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 20px; border-radius: 15px; margin: 10px 0;">
    <span style="font-size: 2em;">❌</span>
    <strong>Recommendations Error:</strong> {str(e)}
</div>
"""
    
    def export_chat_history(self):
        """Export chat history to JSON"""
        if not self.chat_history:
            return None
        
        export_path = os.path.join(LOGS_DIR, f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(export_path, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
        
        logger_app.info(f"Chat history exported to {export_path}")
        return export_path

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_gradio_app():
    """Create Gradio interface"""
    
    app = HermesApp()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Hermes AI Logistics Assistant") as demo:
        
        gr.Markdown("""
        <div class="main-header">🚀 Hermes AI Logistics Assistant</div>
        <p style="text-align: center; color: #666; font-size: 1.2em;">
            Powered by PandasAI + Local LLM | Ask anything about your shipments in natural language
        </p>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Data Source")
                
                data_input_mode = gr.Radio(
                    ["Select Existing Data CSV", "Upload New Data CSV"],
                    label="Data Source Mode",
                    value="Select Existing Data CSV"
                )
                
                data_dropdown = gr.Dropdown(
                    choices=app.get_csv_files(),
                    label="Select Data File",
                    value=SHIPMENTS_FILE if os.path.exists(SHIPMENTS_FILE) else None,
                    visible=True
                )
                
                data_uploader = gr.File(
                    label="Upload CSV",
                    file_types=[".csv"],
                    visible=False
                )
                
                def switch_mode(mode):
                    if mode == "Upload New Data CSV":
                        return gr.Dropdown(visible=False), gr.File(visible=True)
                    return gr.Dropdown(visible=True), gr.File(visible=False)
                
                data_input_mode.change(
                    fn=switch_mode,
                    inputs=data_input_mode,
                    outputs=[data_dropdown, data_uploader]
                )
                
                gr.Markdown("### 💡 Sample Questions")
                question_dropdown = gr.Dropdown(
                    choices=app.get_questions(),
                    label="Or select a sample question",
                    allow_custom_value=True
                )
                
                gr.Markdown("### 📊 Quick Actions")
                predict_btn = gr.Button("🔮 Get Predictions", variant="secondary")
                recommend_btn = gr.Button("💡 Get Recommendations", variant="secondary")
                export_btn = gr.Button("💾 Export Chat History", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Hermes Anything")
                
                user_prompt = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the main causes of delays and how can we reduce them?",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Analyze", variant="primary", scale=3)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)
                
                response_output = gr.Markdown(label="Response")
                
                with gr.Tabs():
                    with gr.Tab("📈 Visualization"):
                        chart_output = gr.Image(label="Chart", height=400)
                    
                    with gr.Tab("📊 Statistics"):
                        stats_output = gr.JSON(label="Summary Statistics")
                    
                    with gr.Tab("🗃️ Data Preview"):
                        data_preview = gr.Dataframe(label="Data Sample")
                    
                    with gr.Tab("💬 Chat History"):
                        history_output = gr.JSON(label="Conversation History")
        
        # Event handlers
        submit_btn.click(
            fn=app.process_query,
            inputs=[data_input_mode, data_uploader, data_dropdown, question_dropdown, user_prompt],
            outputs=[response_output, chart_output, stats_output, data_preview, history_output]
        )
        
        predict_btn.click(
            fn=app.get_predictions,
            outputs=response_output
        )
        
        recommend_btn.click(
            fn=app.get_recommendations,
            outputs=response_output
        )
        
        export_btn.click(
            fn=app.export_chat_history,
            outputs=gr.File(label="Download")
        )
        
        clear_btn.click(
            fn=lambda: ("", None, None, None, []),
            outputs=[response_output, chart_output, stats_output, data_preview, history_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666; padding: 20px;">
            <p><strong>Hermes AI</strong> | Built with PandasAI, Qwen LLM, Plotly, and Gradio</p>
            <p>💡 Ask complex questions, get intelligent insights, and optimize your logistics operations</p>
        </div>
        """)
    
    return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger_app.info("Starting Hermes AI Logistics Assistant...")
    
    demo = create_gradio_app()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )