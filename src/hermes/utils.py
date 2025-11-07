import os
import logging
import pandas as pd
from typing import List, Dict
from datetime import datetime
import random

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