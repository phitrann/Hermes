#!/usr/bin/env python
"""
Explainability Evaluation Suite for Hermes AI

Tests the transparency and interpretability of system reasoning.
Evaluates:
- Reasoning visibility (classification shown)
- Data source attribution (references to columns/data)
- Decision justification quality (explanation depth)
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from hermes.app import HermesApp
from hermes.ui_agent import LLMReasoningCapture
import logging


class ExplainabilityEvaluator:
    """Evaluates explainability of Hermes responses."""
    
    def __init__(self, data_path: str = "data/shipments.csv"):
        """Initialize evaluator with test data."""
        self.app = HermesApp()
        self.df = pd.read_csv(data_path)
        self.app.load_data(self.df)
        
        # Set up reasoning capture
        self.reasoning_capture = LLMReasoningCapture()
        logging.getLogger().addHandler(self.reasoning_capture)
        
        # Test results
        self.results = {
            "total_queries": 0,
            "reasoning_visibility_count": 0,
            "data_attribution_count": 0,
            "justification_scores": [],
            "test_cases": []
        }
    
    def extract_confidence(self, reasoning: str) -> float:
        """Extract confidence score from reasoning text."""
        # Look for patterns like "confidence: 0.92" or "0.92 confidence"
        matches = re.findall(r'confidence[:\s]+(\d+\.?\d*)', reasoning, re.IGNORECASE)
        if matches:
            return float(matches[0])
        
        # Look for standalone decimals between 0 and 1
        matches = re.findall(r'\b(0\.\d+)\b', reasoning)
        if matches:
            return float(matches[0])
        
        return 0.0
    
    def check_reasoning_visibility(self, reasoning: str) -> bool:
        """Check if reasoning contains classification and confidence."""
        keywords = ["classified", "intent", "confidence"]
        return all(keyword in reasoning.lower() for keyword in keywords)
    
    def check_data_attribution(self, reasoning: str, response: str) -> bool:
        """Check if response references specific data sources."""
        # Look for column names
        data_columns = ["warehouse", "route", "delay", "shipment", "date", "carrier"]
        
        combined_text = (reasoning + " " + response).lower()
        
        # Check for at least 2 data column references
        references = sum(1 for col in data_columns if col in combined_text)
        
        # Also check for aggregation method mentions
        methods = ["average", "mean", "count", "sum", "grouped", "filtered"]
        method_mentions = sum(1 for method in methods if method in combined_text)
        
        return references >= 2 or method_mentions >= 1
    
    def score_justification(self, reasoning: str, response: str) -> int:
        """Score decision justification quality (1-5 scale)."""
        combined_text = reasoning + " " + response
        score = 1
        
        # Level 2: Mentions method used
        if any(word in combined_text.lower() for word in ["analyzed", "calculated", "computed", "evaluated"]):
            score = 2
        
        # Level 3: Describes logic
        if any(word in combined_text.lower() for word in ["grouping", "filtering", "aggregating", "comparing"]):
            score = 3
        
        # Level 4: Detailed explanation with data sources
        data_refs = sum(1 for col in ["warehouse", "route", "delay", "shipment"] if col in combined_text.lower())
        if data_refs >= 2 and score >= 3:
            score = 4
        
        # Level 5: Comprehensive with step-by-step logic
        if "step" in combined_text.lower() or combined_text.count("\n") >= 5:
            score = 5
        
        return score
    
    def test_query_with_reasoning(self, query: str, expected_intent: str = None) -> Dict[str, Any]:
        """Test a single query and evaluate its explainability."""
        # Clear previous reasoning
        self.reasoning_capture.clear()
        
        # Process query
        try:
            response_html, chart_path, stats, preview_df, history = self.app.process_query_chat(query, [])
            
            # Get captured reasoning
            reasoning = self.reasoning_capture.get_reasoning_markdown()
            
            # Evaluate reasoning
            has_reasoning = self.check_reasoning_visibility(reasoning)
            has_attribution = self.check_data_attribution(reasoning, response_html)
            justification_score = self.score_justification(reasoning, response_html)
            confidence = self.extract_confidence(reasoning)
            
            result = {
                "query": query,
                "has_reasoning_visibility": has_reasoning,
                "has_data_attribution": has_attribution,
                "justification_score": justification_score,
                "confidence_score": confidence,
                "reasoning_text": reasoning[:300],  # First 300 chars
                "response_text": response_html[:300],
                "expected_intent": expected_intent
            }
            
            # Update counters
            if has_reasoning:
                self.results["reasoning_visibility_count"] += 1
            if has_attribution:
                self.results["data_attribution_count"] += 1
            self.results["justification_scores"].append(justification_score)
            
            return result
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "has_reasoning_visibility": False,
                "has_data_attribution": False,
                "justification_score": 0
            }
    
    def test_statistics_queries(self) -> List[Dict[str, Any]]:
        """Test explainability of statistics queries."""
        print("\n📊 Testing Statistics Query Explainability...")
        
        queries = [
            ("How many shipments were delayed?", "statistics"),
            ("What is the average delay?", "statistics"),
            ("Show me summary statistics", "statistics")
        ]
        
        results = []
        for query, expected_intent in queries:
            result = self.test_query_with_reasoning(query, expected_intent)
            results.append(result)
            
            status = "✅" if result.get("has_reasoning_visibility") else "❌"
            print(f"  {status} {query[:50]}... (Justification: {result.get('justification_score', 0)}/5)")
        
        return results
    
    def test_visualization_queries(self) -> List[Dict[str, Any]]:
        """Test explainability of visualization queries."""
        print("\n📊 Testing Visualization Query Explainability...")
        
        queries = [
            ("Show me delays by warehouse", "visualization"),
            ("Create a chart of shipment trends", "visualization"),
            ("Visualize route performance", "visualization")
        ]
        
        results = []
        for query, expected_intent in queries:
            result = self.test_query_with_reasoning(query, expected_intent)
            results.append(result)
            
            status = "✅" if result.get("has_reasoning_visibility") else "❌"
            print(f"  {status} {query[:50]}... (Justification: {result.get('justification_score', 0)}/5)")
        
        return results
    
    def test_prediction_queries(self) -> List[Dict[str, Any]]:
        """Test explainability of prediction queries."""
        print("\n📊 Testing Prediction Query Explainability...")
        
        queries = [
            ("Predict delays for next week", "prediction"),
            ("What delays should I expect?", "prediction"),
            ("Forecast future shipment performance", "prediction")
        ]
        
        results = []
        for query, expected_intent in queries:
            result = self.test_query_with_reasoning(query, expected_intent)
            results.append(result)
            
            status = "✅" if result.get("has_reasoning_visibility") else "❌"
            print(f"  {status} {query[:50]}... (Justification: {result.get('justification_score', 0)}/5)")
        
        return results
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete explainability evaluation."""
        print("=" * 60)
        print("🔍 Hermes AI - Explainability Evaluation Suite")
        print("=" * 60)
        
        # Run test categories
        stats_results = self.test_statistics_queries()
        viz_results = self.test_visualization_queries()
        pred_results = self.test_prediction_queries()
        
        # Combine all results
        all_results = stats_results + viz_results + pred_results
        self.results["test_cases"] = all_results
        self.results["total_queries"] = len(all_results)
        
        # Calculate metrics
        reasoning_visibility_rate = (
            self.results["reasoning_visibility_count"] / self.results["total_queries"]
            if self.results["total_queries"] > 0 else 0.0
        )
        
        data_attribution_rate = (
            self.results["data_attribution_count"] / self.results["total_queries"]
            if self.results["total_queries"] > 0 else 0.0
        )
        
        avg_justification_score = (
            sum(self.results["justification_scores"]) / len(self.results["justification_scores"])
            if self.results["justification_scores"] else 0.0
        )
        
        # Calculate overall explainability score
        explainability_score = (
            reasoning_visibility_rate * 0.3 +
            data_attribution_rate * 0.3 +
            (avg_justification_score / 5.0) * 0.4
        )
        
        # Update results
        self.results.update({
            "reasoning_visibility_rate": reasoning_visibility_rate,
            "data_attribution_rate": data_attribution_rate,
            "avg_justification_score": avg_justification_score,
            "overall_explainability_score": explainability_score,
            "evaluation_date": datetime.now().isoformat(),
            "status": "PASS" if explainability_score >= 0.75 else "FAIL"
        })
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Explainability Evaluation Summary")
        print("=" * 60)
        print(f"Total Queries: {self.results['total_queries']}")
        print(f"Reasoning Visibility: {reasoning_visibility_rate:.2%}")
        print(f"Data Attribution: {data_attribution_rate:.2%}")
        print(f"Avg Justification Score: {avg_justification_score:.2f}/5.0")
        print(f"Overall Explainability Score: {explainability_score:.2%}")
        print(f"\nStatus: {self.results['status']}")
        
        # Show sample reasoning
        if all_results:
            print("\n📝 Sample Reasoning Chain:")
            sample = all_results[0]
            print(f"Query: {sample['query']}")
            print(f"Reasoning: {sample.get('reasoning_text', 'N/A')[:200]}...")
        
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_path: str = "evaluation/results/explainability_report.json"):
        """Save evaluation results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Run explainability evaluation."""
    import os
    mock_mode = os.getenv("HERMES_TEST_MOCK", "false").lower() == "true"
    
    if mock_mode:
        print("⚠️  Running in MOCK mode (no LLM calls)")
        print("Set HERMES_TEST_MOCK=false to run with actual LLM")
        return
    
    evaluator = ExplainabilityEvaluator()
    results = evaluator.run_full_evaluation()
    evaluator.save_results()
    
    # Exit with appropriate code
    exit_code = 0 if results["status"] == "PASS" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
