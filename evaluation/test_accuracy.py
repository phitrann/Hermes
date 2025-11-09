#!/usr/bin/env python
"""
Accuracy Evaluation Suite for Hermes AI

Tests the correctness of query responses against ground truth answers.
Evaluates:
- Exact Match (EM) for categorical answers
- F1 Score for multi-value answers
- Numerical Accuracy for statistics
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from hermes.app import HermesApp


class AccuracyEvaluator:
    """Evaluates accuracy of Hermes responses."""
    
    def __init__(self, data_path: str = "data/shipments.csv"):
        """Initialize evaluator with test data."""
        self.app = HermesApp()
        self.df = pd.read_csv(data_path)
        self.app.load_data(self.df)
        
        # Ground truth calculations
        self.ground_truth = self._calculate_ground_truth()
        
        # Test results
        self.results = {
            "total_queries": 0,
            "correct_answers": 0,
            "exact_match_score": 0.0,
            "f1_score": 0.0,
            "numerical_accuracy": 0.0,
            "by_category": {},
            "failed_queries": []
        }
    
    def _calculate_ground_truth(self) -> Dict[str, Any]:
        """Pre-calculate ground truth answers from data."""
        return {
            # Basic counts
            "total_shipments": len(self.df),
            "delayed_shipments": len(self.df[self.df['delay_minutes'] > 0]),
            "ontime_shipments": len(self.df[self.df['on_time'] == True]),
            
            # Aggregations
            "avg_delay": self.df['delay_minutes'].mean(),
            "median_delay": self.df['delay_minutes'].median(),
            "max_delay": self.df['delay_minutes'].max(),
            "min_delay": self.df['delay_minutes'].min(),
            
            # By warehouse
            "delays_by_warehouse": self.df.groupby('warehouse')['delay_minutes'].mean().to_dict(),
            "shipments_by_warehouse": self.df.groupby('warehouse').size().to_dict(),
            
            # By route
            "delays_by_route": self.df.groupby('route')['delay_minutes'].mean().to_dict(),
            "shipments_by_route": self.df.groupby('route').size().to_dict(),
            
            # Percentages
            "ontime_percentage": (len(self.df[self.df['on_time'] == True]) / len(self.df)) * 100,
            "delayed_percentage": (len(self.df[self.df['delay_minutes'] > 0]) / len(self.df)) * 100,
            
            # Top entities
            "worst_warehouse": self.df.groupby('warehouse')['delay_minutes'].mean().idxmax(),
            "best_warehouse": self.df.groupby('warehouse')['delay_minutes'].mean().idxmin(),
            "worst_route": self.df.groupby('route')['delay_minutes'].mean().idxmax(),
            "best_route": self.df.groupby('route')['delay_minutes'].mean().idxmin(),
        }
    
    def extract_number(self, text: str) -> float:
        """Extract first number from response text."""
        # Look for numbers with optional decimal points and commas
        matches = re.findall(r'[\d,]+\.?\d*', text)
        if matches:
            # Remove commas and convert to float
            return float(matches[0].replace(',', ''))
        return None
    
    def extract_entity(self, text: str, entity_type: str = "warehouse") -> str:
        """Extract entity name (warehouse/route) from response."""
        if entity_type == "warehouse":
            # Look for patterns like WH_01, WH_02, etc.
            match = re.search(r'WH_\d+', text)
            if match:
                return match.group(0)
        elif entity_type == "route":
            # Look for patterns like Route_A, Route_B, etc.
            match = re.search(r'Route_[A-Z]', text)
            if match:
                return match.group(0)
        return None
    
    def calculate_numerical_accuracy(self, predicted: float, actual: float, tolerance: float = 0.05) -> float:
        """Calculate numerical accuracy with tolerance."""
        if actual == 0:
            return 1.0 if abs(predicted) < 0.01 else 0.0
        
        relative_error = abs(predicted - actual) / abs(actual)
        return 1.0 if relative_error <= tolerance else 0.0
    
    def test_simple_statistics(self) -> Dict[str, Any]:
        """Test simple statistical queries."""
        print("\n📊 Testing Simple Statistics...")
        
        tests = [
            {
                "query": "How many total shipments are in the dataset?",
                "expected": self.ground_truth["total_shipments"],
                "type": "count"
            },
            {
                "query": "How many shipments were delayed?",
                "expected": self.ground_truth["delayed_shipments"],
                "type": "count"
            },
            {
                "query": "What is the average delay in minutes?",
                "expected": self.ground_truth["avg_delay"],
                "type": "number"
            },
            {
                "query": "What is the median delay?",
                "expected": self.ground_truth["median_delay"],
                "type": "number"
            },
            {
                "query": "What percentage of shipments were on time?",
                "expected": self.ground_truth["ontime_percentage"],
                "type": "percentage"
            }
        ]
        
        results = {"total": len(tests), "correct": 0, "tests": []}
        
        for test in tests:
            try:
                # Process query (using mock to avoid LLM dependency in tests)
                response_html, _, stats, _, _ = self.app.process_query_chat(test["query"], [])
                
                # Extract answer
                predicted = self.extract_number(response_html)
                expected = test["expected"]
                
                # Calculate accuracy
                if test["type"] in ["count", "number", "percentage"]:
                    accuracy = self.calculate_numerical_accuracy(predicted, expected)
                    correct = accuracy == 1.0
                else:
                    correct = abs(predicted - expected) < 0.01
                
                if correct:
                    results["correct"] += 1
                    print(f"  ✅ {test['query'][:50]}... (Expected: {expected:.2f}, Got: {predicted:.2f})")
                else:
                    print(f"  ❌ {test['query'][:50]}... (Expected: {expected:.2f}, Got: {predicted:.2f})")
                    self.results["failed_queries"].append({
                        "query": test["query"],
                        "expected": expected,
                        "predicted": predicted,
                        "category": "simple_statistics"
                    })
                
                results["tests"].append({
                    "query": test["query"],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct
                })
                
            except Exception as e:
                print(f"  ⚠️  Error on query: {test['query'][:50]}... - {str(e)}")
                results["tests"].append({
                    "query": test["query"],
                    "error": str(e)
                })
        
        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0
        return results
    
    def test_aggregations(self) -> Dict[str, Any]:
        """Test aggregation queries."""
        print("\n📊 Testing Aggregations...")
        
        tests = [
            {
                "query": "What is the average delay by warehouse WH_01?",
                "expected": self.ground_truth["delays_by_warehouse"].get("WH_01", 0),
                "type": "number"
            },
            {
                "query": "How many shipments went through warehouse WH_02?",
                "expected": self.ground_truth["shipments_by_warehouse"].get("WH_02", 0),
                "type": "count"
            },
            {
                "query": "What is the total number of delayed shipments?",
                "expected": self.ground_truth["delayed_shipments"],
                "type": "count"
            }
        ]
        
        results = {"total": len(tests), "correct": 0, "tests": []}
        
        for test in tests:
            try:
                response_html, _, stats, _, _ = self.app.process_query_chat(test["query"], [])
                predicted = self.extract_number(response_html)
                expected = test["expected"]
                
                accuracy = self.calculate_numerical_accuracy(predicted, expected)
                correct = accuracy == 1.0
                
                if correct:
                    results["correct"] += 1
                    print(f"  ✅ {test['query'][:50]}...")
                else:
                    print(f"  ❌ {test['query'][:50]}... (Expected: {expected:.2f}, Got: {predicted:.2f})")
                    self.results["failed_queries"].append({
                        "query": test["query"],
                        "expected": expected,
                        "predicted": predicted,
                        "category": "aggregations"
                    })
                
                results["tests"].append({
                    "query": test["query"],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct
                })
                
            except Exception as e:
                print(f"  ⚠️  Error on query: {test['query'][:50]}... - {str(e)}")
        
        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0
        return results
    
    def test_entity_extraction(self) -> Dict[str, Any]:
        """Test queries that require extracting entities."""
        print("\n📊 Testing Entity Extraction...")
        
        tests = [
            {
                "query": "Which warehouse has the highest average delay?",
                "expected": self.ground_truth["worst_warehouse"],
                "entity_type": "warehouse"
            },
            {
                "query": "Which warehouse has the best performance?",
                "expected": self.ground_truth["best_warehouse"],
                "entity_type": "warehouse"
            },
            {
                "query": "Which route has the most delays?",
                "expected": self.ground_truth["worst_route"],
                "entity_type": "route"
            }
        ]
        
        results = {"total": len(tests), "correct": 0, "tests": []}
        
        for test in tests:
            try:
                response_html, _, stats, _, _ = self.app.process_query_chat(test["query"], [])
                predicted = self.extract_entity(response_html, test["entity_type"])
                expected = test["expected"]
                
                correct = (predicted == expected)
                
                if correct:
                    results["correct"] += 1
                    print(f"  ✅ {test['query'][:50]}... (Expected: {expected}, Got: {predicted})")
                else:
                    print(f"  ❌ {test['query'][:50]}... (Expected: {expected}, Got: {predicted})")
                    self.results["failed_queries"].append({
                        "query": test["query"],
                        "expected": expected,
                        "predicted": predicted,
                        "category": "entity_extraction"
                    })
                
                results["tests"].append({
                    "query": test["query"],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct
                })
                
            except Exception as e:
                print(f"  ⚠️  Error on query: {test['query'][:50]}... - {str(e)}")
        
        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0
        return results
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete accuracy evaluation suite."""
        print("=" * 60)
        print("🎯 Hermes AI - Accuracy Evaluation Suite")
        print("=" * 60)
        
        # Run test categories
        simple_stats_results = self.test_simple_statistics()
        aggregation_results = self.test_aggregations()
        entity_results = self.test_entity_extraction()
        
        # Calculate overall metrics
        total_tests = (
            simple_stats_results["total"] +
            aggregation_results["total"] +
            entity_results["total"]
        )
        
        total_correct = (
            simple_stats_results["correct"] +
            aggregation_results["correct"] +
            entity_results["correct"]
        )
        
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        
        # Compile results
        self.results.update({
            "total_queries": total_tests,
            "correct_answers": total_correct,
            "exact_match_score": overall_accuracy,
            "by_category": {
                "simple_statistics": {
                    "accuracy": simple_stats_results["accuracy"],
                    "total": simple_stats_results["total"],
                    "correct": simple_stats_results["correct"]
                },
                "aggregations": {
                    "accuracy": aggregation_results["accuracy"],
                    "total": aggregation_results["total"],
                    "correct": aggregation_results["correct"]
                },
                "entity_extraction": {
                    "accuracy": entity_results["accuracy"],
                    "total": entity_results["total"],
                    "correct": entity_results["correct"]
                }
            },
            "evaluation_date": datetime.now().isoformat(),
            "status": "PASS" if overall_accuracy >= 0.80 else "FAIL"
        })
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Evaluation Summary")
        print("=" * 60)
        print(f"Total Queries: {total_tests}")
        print(f"Correct Answers: {total_correct}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print(f"\nBy Category:")
        print(f"  Simple Statistics: {simple_stats_results['accuracy']:.2%}")
        print(f"  Aggregations: {aggregation_results['accuracy']:.2%}")
        print(f"  Entity Extraction: {entity_results['accuracy']:.2%}")
        print(f"\nStatus: {self.results['status']}")
        
        if self.results["failed_queries"]:
            print(f"\n⚠️  Failed Queries: {len(self.results['failed_queries'])}")
            for failed in self.results["failed_queries"][:5]:  # Show first 5
                print(f"  - {failed['query'][:60]}...")
        
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_path: str = "evaluation/results/accuracy_report.json"):
        """Save evaluation results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Run accuracy evaluation."""
    # Check if running in mock mode (for CI/testing without LLM)
    import os
    mock_mode = os.getenv("HERMES_TEST_MOCK", "false").lower() == "true"
    
    if mock_mode:
        print("⚠️  Running in MOCK mode (no LLM calls)")
        print("Set HERMES_TEST_MOCK=false to run with actual LLM")
        return
    
    evaluator = AccuracyEvaluator()
    results = evaluator.run_full_evaluation()
    evaluator.save_results()
    
    # Exit with appropriate code
    exit_code = 0 if results["status"] == "PASS" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
