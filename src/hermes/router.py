"""
LLM-driven Query Router
Uses SmartDataframe.chat to classify intent, falls back to regex.
"""
import re
import json
import logging
from typing import Dict, Any
from .prompts import PROMPT_TEMPLATES
from pandasai import SmartDataframe

logger = logging.getLogger(__name__)

class QueryRouter:
    def __init__(self):
        self.intent_patterns = {
            'prediction': [r'\b(predict|forecast|future|next week|next month|what will|will be)\b'],
            'recommendation': [r'\b(recommend|recommendation|suggestion|suggest|how to|improve|optimize)\b'],
            'visualization': [r'\b(visualize|chart|graph|plot|show me|display)\b'],
            'statistics': [r'\b(statistics|stats|summary|overview|metrics)\b', r'\b(how many|count|total|average|mean)\b'],
            'comparison': [r'\b(compare|comparison|versus| vs |best|worst)\b']
        }
        self.allowed_intents = ['prediction', 'recommendation', 'visualization', 'comparison', 'statistics', 'general']

    def classify_query(self, query: str, smart_df: SmartDataframe = None) -> Dict[str, Any]:
        # Try LLM-based classification
        if smart_df is not None:
            try:
                prompt = PROMPT_TEMPLATES['classification_intent'].format(query=query)
                llm_response = smart_df.chat(prompt)
                text = str(llm_response).strip()
                json_str = None
                if text.startswith('{'):
                    json_str = text
                else:
                    start = text.find('{')
                    end = text.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        json_str = text[start:end+1]
                if json_str:
                    parsed = json.loads(json_str)
                    intent = parsed.get('intent', 'general').lower()
                    confidence = parsed.get('confidence', None)
                    if intent in self.allowed_intents:
                        try:
                            confidence = float(confidence) if confidence is not None else 0.85
                            confidence = max(0.0, min(1.0, confidence))
                        except Exception:
                            confidence = 0.85
                        logger.info(f"LLM router: intent={intent}, confidence={confidence}")
                        return {'intent': intent, 'confidence': confidence, 'method': 'llm'}
                logger.warning("LLM did not return a valid JSON intent; falling back to regex.")
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")

        # Fallback regex-based classification
        q = query.lower()
        for intent, patterns in self.intent_patterns.items():
            for p in patterns:
                if re.search(p, q):
                    logger.info(f"Regex router: classified intent='{intent}'")
                    return {'intent': intent, 'confidence': 0.6, 'method': 'fallback'}
        return {'intent': 'general', 'confidence': 0.5, 'method': 'fallback'}

    def should_force_chart(self, query: str, smart_df: SmartDataframe = None) -> bool:
        r = self.classify_query(query, smart_df=smart_df)
        return r['intent'] in ['visualization', 'comparison']