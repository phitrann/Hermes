#!/usr/bin/env python
"""
Performance Evaluation Suite for Hermes AI

Tests system response time and efficiency.
Evaluates:
- Data loading time
- Query classification time
- LLM processing time
- Chart generation time
- Total end-to-end response time
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from hermes.app import HermesApp
from hermes.router import QueryRouter


class PerformanceEvaluator:
    """Evaluates performance of Hermes system."""
    
    def __init__(self, data_path: str = "data/shipments.csv"):
        """Initialize evaluator with test data."""
        self.data_path = data_path
        self.app = None  # Will be initialized per test
        
        # Performance targets (in seconds)
        self.targets = {
            "data_loading": {"mean": 3.0, "p95": 5.0},
            "classification": {"mean": 1.0, "p95": 2.0},
            "llm_processing": {"mean": 3.0, "p95": 5.0},
            "chart_generation": {"mean": 4.0, "p95": 7.0},
            "total_text_query": {"mean": 5.0, "p95": 8.0},
            "total_chart_query": {"mean": 8.0, "p95": 12.0}
        }
        
        # Test results
        self.results = {
            "data_loading": {},
            "classification": {},
            "text_queries": {},
            "chart_queries": {},
            "evaluation_date": datetime.now().isoformat()
        }
    
    def measure_data_loading(self, iterations: int = 5) -> Dict[str, float]:
        """Measure data loading performance."""
        print("\n⏱️  Measuring Data Loading Time...")
        
        times = []
        
        for i in range(iterations):
            # Create fresh app instance
            app = HermesApp()
            
            start = time.perf_counter()
            df = pd.read_csv(self.data_path)
            app.load_data(df)
            end = time.perf_counter()
            
            elapsed = end - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.3f}s")
        
        metrics = self._calculate_metrics(times, "data_loading")
        self._print_metrics("Data Loading", metrics)
        
        return metrics
    
    def measure_classification(self, iterations: int = 10) -> Dict[str, float]:
        """Measure query classification performance."""
        print("\n⏱️  Measuring Classification Time...")
        
        # Initialize app once
        self.app = HermesApp()
        df = pd.read_csv(self.data_path)
        self.app.load_data(df)
        
        test_queries = [
            "Show me delays by warehouse",
            "Predict delays for next week",
            "Give me recommendations",
            "What is the average delay?",
            "Compare route performance"
        ]
        
        times = []
        
        for i in range(iterations):
            query = test_queries[i % len(test_queries)]
            
            start = time.perf_counter()
            result = self.app.router.classify_query(query, self.app.smart_df)
            end = time.perf_counter()
            
            elapsed = end - start
            times.append(elapsed)
            print(f"  Query {i+1}: {elapsed:.3f}s - Intent: {result.get('intent', 'unknown')}")
        
        metrics = self._calculate_metrics(times, "classification")
        self._print_metrics("Classification", metrics)
        
        return metrics
    
    def measure_text_queries(self, iterations: int = 10) -> Dict[str, float]:
        """Measure text-only query performance."""
        print("\n⏱️  Measuring Text Query Performance...")
        
        # Initialize app if not already done
        if not self.app:
            self.app = HermesApp()
            df = pd.read_csv(self.data_path)
            self.app.load_data(df)
        
        test_queries = [
            "How many shipments were delayed?",
            "What is the average delay?",
            "Which warehouse has the best performance?",
            "What percentage of shipments are on time?",
            "How many total shipments are there?"
        ]
        
        times = []
        
        for i in range(iterations):
            query = test_queries[i % len(test_queries)]
            
            start = time.perf_counter()
            try:
                response_html, chart_path, stats, preview_df, history = self.app.process_query_chat(query, [])
                end = time.perf_counter()
                
                elapsed = end - start
                times.append(elapsed)
                has_chart = "✓" if chart_path else "✗"
                print(f"  Query {i+1}: {elapsed:.3f}s [Chart: {has_chart}]")
            except Exception as e:
                print(f"  Query {i+1}: ERROR - {str(e)[:50]}")
        
        if times:
            metrics = self._calculate_metrics(times, "total_text_query")
            self._print_metrics("Text Queries", metrics)
            return metrics
        else:
            return {}
    
    def measure_chart_queries(self, iterations: int = 5) -> Dict[str, float]:
        """Measure chart generation query performance."""
        print("\n⏱️  Measuring Chart Query Performance...")
        
        # Initialize app if not already done
        if not self.app:
            self.app = HermesApp()
            df = pd.read_csv(self.data_path)
            self.app.load_data(df)
        
        test_queries = [
            "Show me delays by warehouse as a chart",
            "Visualize shipment trends over time",
            "Create a chart of route performance",
            "Plot delay distribution",
            "Show me warehouse comparison chart"
        ]
        
        times = []
        
        for i in range(iterations):
            query = test_queries[i % len(test_queries)]
            
            start = time.perf_counter()
            try:
                response_html, chart_path, stats, preview_df, history = self.app.process_query_chat(query, [])
                end = time.perf_counter()
                
                elapsed = end - start
                times.append(elapsed)
                has_chart = "✓" if chart_path else "✗"
                print(f"  Query {i+1}: {elapsed:.3f}s [Chart: {has_chart}]")
            except Exception as e:
                print(f"  Query {i+1}: ERROR - {str(e)[:50]}")
        
        if times:
            metrics = self._calculate_metrics(times, "total_chart_query")
            self._print_metrics("Chart Queries", metrics)
            return metrics
        else:
            return {}
    
    def _calculate_metrics(self, times: List[float], category: str) -> Dict[str, float]:
        """Calculate performance metrics from timing data."""
        if not times:
            return {}
        
        metrics = {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "sample_count": len(times)
        }
        
        # Add target comparison
        if category in self.targets:
            target = self.targets[category]
            metrics["target_mean"] = target["mean"]
            metrics["target_p95"] = target["p95"]
            metrics["meets_mean_target"] = metrics["mean"] <= target["mean"]
            metrics["meets_p95_target"] = metrics["p95"] <= target["p95"]
        
        return metrics
    
    def _print_metrics(self, label: str, metrics: Dict[str, float]):
        """Print formatted metrics."""
        if not metrics:
            print(f"  ⚠️  No metrics available for {label}")
            return
        
        print(f"\n  📊 {label} Metrics:")
        print(f"    Mean: {metrics['mean']:.3f}s")
        print(f"    Median: {metrics['median']:.3f}s")
        print(f"    P95: {metrics['p95']:.3f}s")
        print(f"    Min/Max: {metrics['min']:.3f}s / {metrics['max']:.3f}s")
        
        if "target_mean" in metrics:
            mean_status = "✅" if metrics["meets_mean_target"] else "❌"
            p95_status = "✅" if metrics["meets_p95_target"] else "❌"
            print(f"    {mean_status} Mean Target: {metrics['target_mean']:.3f}s")
            print(f"    {p95_status} P95 Target: {metrics['target_p95']:.3f}s")
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete performance evaluation."""
        print("=" * 60)
        print("⚡ Hermes AI - Performance Evaluation Suite")
        print("=" * 60)
        
        # Run benchmarks
        self.results["data_loading"] = self.measure_data_loading(iterations=3)
        self.results["classification"] = self.measure_classification(iterations=10)
        self.results["text_queries"] = self.measure_text_queries(iterations=10)
        self.results["chart_queries"] = self.measure_chart_queries(iterations=5)
        
        # Calculate overall status
        all_targets_met = all(
            metrics.get("meets_mean_target", False) and metrics.get("meets_p95_target", False)
            for metrics in self.results.values()
            if isinstance(metrics, dict) and "meets_mean_target" in metrics
        )
        
        self.results["overall_status"] = "PASS" if all_targets_met else "PARTIAL_PASS"
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Performance Evaluation Summary")
        print("=" * 60)
        
        for category, metrics in self.results.items():
            if isinstance(metrics, dict) and "mean" in metrics:
                status = "✅" if metrics.get("meets_mean_target", False) else "⚠️"
                print(f"{status} {category.replace('_', ' ').title()}: {metrics['mean']:.3f}s (target: {metrics.get('target_mean', 'N/A')}s)")
        
        print(f"\nOverall Status: {self.results['overall_status']}")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_path: str = "evaluation/results/performance_metrics.json"):
        """Save evaluation results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Run performance evaluation."""
    import os
    mock_mode = os.getenv("HERMES_TEST_MOCK", "false").lower() == "true"
    
    if mock_mode:
        print("⚠️  Running in MOCK mode (no LLM calls)")
        print("Set HERMES_TEST_MOCK=false to run with actual LLM")
        return
    
    evaluator = PerformanceEvaluator()
    results = evaluator.run_full_evaluation()
    evaluator.save_results()
    
    # Always exit 0 for performance tests (informational only)
    sys.exit(0)


if __name__ == "__main__":
    main()
