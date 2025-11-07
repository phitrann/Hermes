import os
import logging
import pandas as pd
from typing import Iterator, Optional, List, Dict, Any, Union
from datetime import datetime
import traceback
import random

# Import app types
from .models import (
    BaseResponse,
    PredictionResponse,
    RecommendationResponse,
    VisualizationResponse,
    StatisticsResponse,
    GeneralResponse,
    PYDANTIC_V2,
)

logger = logging.getLogger(__name__)


def load_questions_dataset(data_dir: str, filename: str = "questions.csv") -> List[str]:
    """
    Load questions from CSV file.
    
    Args:
        data_dir: Directory containing the questions file
        filename: Name of the questions CSV file (default: questions.csv)
        
    Returns:
        List of question strings
    """
    try:
        # Try multiple possible filenames
        possible_files = [
            filename,
            "questions.csv",
            "shipment_questions.csv",
            "shipment_questions_500.csv",
        ]
        
        questions_path = None
        for fname in possible_files:
            test_path = os.path.join(data_dir, fname)
            if os.path.exists(test_path):
                questions_path = test_path
                break
        
        if questions_path is None:
            logger.warning(f"No questions file found in {data_dir}")
            # Return fallback questions
            return [
                "Show me the distribution of delays by reason",
                "What's the average delay in minutes?",
                "Visualize delay trends over time",
                "Which routes have the most delays?",
                "Compare warehouse performance",
                "Predict next week's delays",
                "Give me recommendations",
                "Show summary statistics",
            ]
        
        # Try different possible column names
        df = pd.read_csv(questions_path)
        
        # Look for question column (try common names)
        question_col = None
        for col in ['question', 'Question', 'query', 'Query', 'text', 'Text']:
            if col in df.columns:
                question_col = col
                break
        
        if question_col is None:
            # Use first column if no standard name found
            question_col = df.columns[0]
            logger.warning(f"Using first column '{question_col}' as questions")
        
        questions = df[question_col].dropna().astype(str).tolist()
        
        # Remove duplicates while preserving order
        seen = set()
        unique_questions = []
        for q in questions:
            q_stripped = q.strip()
            if q_stripped and q_stripped not in seen:
                seen.add(q_stripped)
                unique_questions.append(q_stripped)
        
        if not unique_questions:
            raise ValueError("No valid questions found in dataset")
        
        logger.info(f"Loaded {len(unique_questions)} unique questions from {questions_path}")
        return unique_questions
        
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        import traceback
        traceback.print_exc()
        # Return fallback questions
        return [
            "Show me the distribution of delays by reason",
            "What's the average delay in minutes?",
            "Visualize delay trends over time",
            "Which routes have the most delays?",
            "Compare warehouse performance",
        ]

def get_random_suggestions(questions: List[str], count: int = 3) -> List[str]:
    """
    Get random questions from the dataset.
    
    Args:
        questions: Full list of questions
        count: Number of suggestions to return
        
    Returns:
        List of randomly selected questions
    """
    if len(questions) <= count:
        return questions
    
    return random.sample(questions, count)
    

def extract_response_components(response: Union[BaseResponse, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract and categorize all components from a response for smart display.
    
    Returns dict with:
        - text: Main text content (markdown)
        - dataframe: DataFrame data (if any)
        - number: Single number value (if any)
        - chart_path: Path to chart image (if any)
        - metrics: Metrics data (if any)
        - recommendations: List of recommendations (if any)
        - stats: Statistics summary (if any)
    """
    components = {
        "text": None,
        "dataframe": None,
        "number": None,
        "chart_path": None,
        "metrics": None,
        "recommendations": None,
        "stats": None,
        "data_type": "text",
    }
    
    try:
        if isinstance(response, BaseResponse):
            # Extract text
            components["text"] = response.text or ""
            
            # Extract chart
            if hasattr(response, 'chart') and response.chart:
                try:
                    if hasattr(response.chart, 'path') and response.chart.path:
                        components["chart_path"] = response.chart.path
                except Exception as e:
                    logger.warning(f"Error extracting chart: {e}")
            
            # Handle specific response types
            if isinstance(response, PredictionResponse):
                components["data_type"] = "prediction"
                if response.metrics:
                    components["metrics"] = {
                        "r2_score": response.metrics.r2_score,
                        "rmse": response.metrics.rmse,
                        "model_type": response.metrics.model_type,
                    }
            
            elif isinstance(response, RecommendationResponse):
                components["data_type"] = "recommendation"
                if response.recommendations:
                    components["recommendations"] = [
                        {
                            "category": r.category,
                            "priority": r.priority,
                            "finding": r.finding,
                            "action": r.action,
                        }
                        for r in response.recommendations
                    ]
            
            elif isinstance(response, StatisticsResponse):
                components["data_type"] = "statistics"
                if response.stats:
                    components["stats"] = {
                        "total_shipments": response.stats.total_shipments,
                        "delayed_shipments": response.stats.delayed_shipments,
                        "on_time_rate": response.stats.delay_rate,
                        "avg_delay_minutes": response.stats.avg_delay_minutes,
                    }
                # Handle DataFrame
                if hasattr(response, 'data_type') and response.data_type == "dataframe":
                    components["data_type"] = "dataframe"
                    if hasattr(response, 'raw_result') and response.raw_result is not None:
                        try:
                            from pandasai.core.response import DataFrameResponse
                            
                            df = None
                            if isinstance(response.raw_result, DataFrameResponse):
                                df = response.raw_result.value
                            elif isinstance(response.raw_result, pd.DataFrame):
                                df = response.raw_result
                            
                            if df is not None and isinstance(df, pd.DataFrame):
                                components["dataframe"] = df.head(50)  # Limit to 50 rows
                        except Exception as e:
                            logger.warning(f"Failed to extract dataframe: {e}")
                # Handle Number
                elif hasattr(response, 'data_type') and response.data_type == "number":
                    components["data_type"] = "number"
                    try:
                        from pandasai.core.response import NumberResponse
                        if isinstance(response.raw_result, NumberResponse):
                            components["number"] = float(response.raw_result.value)
                    except Exception as e:
                        logger.warning(f"Failed to extract number: {e}")
            
            elif isinstance(response, GeneralResponse):
                # Handle DataFrame
                if hasattr(response, 'data_type') and response.data_type == "dataframe":
                    components["data_type"] = "dataframe"
                    if hasattr(response, 'raw_result') and response.raw_result is not None:
                        try:
                            from pandasai.core.response import DataFrameResponse
                            
                            df = None
                            if isinstance(response.raw_result, DataFrameResponse):
                                df = response.raw_result.value
                            elif isinstance(response.raw_result, pd.DataFrame):
                                df = response.raw_result
                            
                            if df is not None and isinstance(df, pd.DataFrame):
                                components["dataframe"] = df.head(50)  # Limit to 50 rows
                        except Exception as e:
                            logger.warning(f"Failed to extract dataframe: {e}")
                
                # Handle Number
                elif hasattr(response, 'data_type') and response.data_type == "number":
                    components["data_type"] = "number"
                    try:
                        from pandasai.core.response import NumberResponse
                        if isinstance(response.raw_result, NumberResponse):
                            components["number"] = float(response.raw_result.value)
                    except Exception as e:
                        logger.warning(f"Failed to extract number: {e}")
        
        else:
            # Fallback for dict response
            components["text"] = response.get("text", "")
            components["chart_path"] = response.get("chart")
    
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error extracting components: {e}", exc_info=True)
        components["text"] = f"Error extracting response components: {str(e)}"
    
    return components    

# ============================================================================
# LLM REASONING CAPTURE (For showing model inner thoughts)
# ============================================================================

class LLMReasoningCapture(logging.Handler):
    """Custom handler to capture LLM reasoning/inner thoughts from logs."""
    
    def __init__(self):
        super().__init__()
        self.reasoning_logs = []
        self.setLevel(logging.DEBUG)
    
    def emit(self, record):
        """Capture log records that contain reasoning/thinking."""
        try:
            msg = record.getMessage()
            # Capture logs from pandasai, litellm, and hermes that contain meaningful content
            # if any(keyword in msg.lower() for keyword in 
            #        ['query', 'analysis', 'response', 'processing', 'detected', 'classified', 'thinking', 'result']):
            self.reasoning_logs.append({
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name.split('.')[-1],  # Get last part of logger name
                'message': msg[:500]  # Increased limit for better context
            })
        except Exception:
            pass
    
    def clear(self):
        """Clear reasoning logs for next query."""
        self.reasoning_logs = []