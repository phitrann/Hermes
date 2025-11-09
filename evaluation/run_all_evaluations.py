#!/usr/bin/env python
"""
Comprehensive Evaluation Suite Runner for Hermes AI

Runs all evaluation tests and generates a consolidated report.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import evaluators
from test_accuracy import AccuracyEvaluator
from test_explainability import ExplainabilityEvaluator
from test_performance import PerformanceEvaluator


class ComprehensiveEvaluator:
    """Runs all evaluation suites and generates consolidated report."""
    
    def __init__(self):
        """Initialize comprehensive evaluator."""
        self.results = {
            "evaluation_date": datetime.now().isoformat(),
            "test_suite_version": "1.0.0",
            "system_info": self._get_system_info(),
            "accuracy": {},
            "explainability": {},
            "performance": {},
            "overall_status": "UNKNOWN"
        }
    
    def _get_system_info(self) -> dict:
        """Collect system information."""
        import platform
        import sys
        
        try:
            from hermes.config import llm, LLM_ENDPOINT, LLM_MODEL
            llm_endpoint = LLM_ENDPOINT
            llm_model = LLM_MODEL
        except:
            llm_endpoint = "Unknown"
            llm_model = "Unknown"
        
        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "llm_endpoint": llm_endpoint,
            "llm_model": llm_model
        }
    
    def run_all_evaluations(self) -> dict:
        """Run all evaluation suites."""
        print("=" * 70)
        print("🚀 Hermes AI - Comprehensive Evaluation Suite")
        print("=" * 70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Run Accuracy Evaluation
        print("\n" + "🎯" * 35)
        print("Running Accuracy Evaluation...")
        print("🎯" * 35)
        try:
            accuracy_eval = AccuracyEvaluator()
            self.results["accuracy"] = accuracy_eval.run_full_evaluation()
            accuracy_eval.save_results()
        except Exception as e:
            print(f"❌ Accuracy evaluation failed: {e}")
            self.results["accuracy"] = {"status": "ERROR", "error": str(e)}
        
        # Run Explainability Evaluation
        print("\n" + "🔍" * 35)
        print("Running Explainability Evaluation...")
        print("🔍" * 35)
        try:
            explainability_eval = ExplainabilityEvaluator()
            self.results["explainability"] = explainability_eval.run_full_evaluation()
            explainability_eval.save_results()
        except Exception as e:
            print(f"❌ Explainability evaluation failed: {e}")
            self.results["explainability"] = {"status": "ERROR", "error": str(e)}
        
        # Run Performance Evaluation
        print("\n" + "⚡" * 35)
        print("Running Performance Evaluation...")
        print("⚡" * 35)
        try:
            performance_eval = PerformanceEvaluator()
            self.results["performance"] = performance_eval.run_full_evaluation()
            performance_eval.save_results()
        except Exception as e:
            print(f"❌ Performance evaluation failed: {e}")
            self.results["performance"] = {"status": "ERROR", "error": str(e)}
        
        # Determine overall status
        self._determine_overall_status()
        
        # Print final summary
        self._print_final_summary()
        
        return self.results
    
    def _determine_overall_status(self):
        """Determine overall evaluation status."""
        accuracy_pass = self.results["accuracy"].get("status") == "PASS"
        explainability_pass = self.results["explainability"].get("status") == "PASS"
        performance_pass = self.results["performance"].get("overall_status") in ["PASS", "PARTIAL_PASS"]
        
        if accuracy_pass and explainability_pass and performance_pass:
            self.results["overall_status"] = "PASS"
        elif accuracy_pass or explainability_pass:
            self.results["overall_status"] = "PARTIAL_PASS"
        else:
            self.results["overall_status"] = "FAIL"
        
        # Generate summary message
        self.results["summary"] = self._generate_summary()
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary."""
        accuracy = self.results["accuracy"]
        explainability = self.results["explainability"]
        performance = self.results["performance"]
        
        lines = []
        
        # Accuracy summary
        if "exact_match_score" in accuracy:
            lines.append(f"Accuracy: {accuracy['exact_match_score']:.1%} ({accuracy['correct_answers']}/{accuracy['total_queries']} queries correct)")
        
        # Explainability summary
        if "overall_explainability_score" in explainability:
            lines.append(f"Explainability: {explainability['overall_explainability_score']:.1%} (avg justification: {explainability.get('avg_justification_score', 0):.1f}/5)")
        
        # Performance summary
        if "text_queries" in performance and "mean" in performance["text_queries"]:
            lines.append(f"Performance: {performance['text_queries']['mean']:.2f}s avg response time")
        
        return " | ".join(lines)
    
    def _print_final_summary(self):
        """Print final evaluation summary."""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 70)
        
        # Overall status
        status_emoji = {
            "PASS": "✅",
            "PARTIAL_PASS": "⚠️",
            "FAIL": "❌",
            "UNKNOWN": "❓"
        }
        emoji = status_emoji.get(self.results["overall_status"], "❓")
        print(f"\n{emoji} Overall Status: {self.results['overall_status']}")
        
        # Individual results
        print("\n📋 Individual Results:")
        print(f"  Accuracy:       {self.results['accuracy'].get('status', 'UNKNOWN')}")
        print(f"  Explainability: {self.results['explainability'].get('status', 'UNKNOWN')}")
        print(f"  Performance:    {self.results['performance'].get('overall_status', 'UNKNOWN')}")
        
        # Summary
        print(f"\n📝 Summary:")
        print(f"  {self.results['summary']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        self._print_recommendations()
        
        print("\n" + "=" * 70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def _print_recommendations(self):
        """Print recommendations based on results."""
        accuracy = self.results["accuracy"]
        explainability = self.results["explainability"]
        performance = self.results["performance"]
        
        # Accuracy recommendations
        if accuracy.get("exact_match_score", 0) < 0.80:
            print("  ⚠️  Accuracy below target (80%) - Review failed queries and improve LLM prompts")
        
        # Explainability recommendations
        if explainability.get("overall_explainability_score", 0) < 0.75:
            print("  ⚠️  Explainability below target (75%) - Enhance reasoning display and data attribution")
        
        # Performance recommendations
        text_query_mean = performance.get("text_queries", {}).get("mean", 0)
        if text_query_mean > 5.0:
            print(f"  ⚠️  Text query response time above target (5s) - Optimize LLM or use caching")
        
        # If everything passes
        if self.results["overall_status"] == "PASS":
            print("  ✅ All evaluation criteria met - System ready for deployment")
    
    def save_consolidated_report(self, output_path: str = "evaluation/results/comprehensive_report.json"):
        """Save consolidated report to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Consolidated report saved to: {output_file}")
        
        # Also save human-readable markdown report
        self._save_markdown_report(output_file.parent / "comprehensive_report.md")
    
    def _save_markdown_report(self, output_path: Path):
        """Save human-readable markdown report."""
        with open(output_path, 'w') as f:
            f.write("# Hermes AI - Comprehensive Evaluation Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall status
            f.write("## Overall Status\n\n")
            f.write(f"**Result:** {self.results['overall_status']}\n\n")
            f.write(f"**Summary:** {self.results['summary']}\n\n")
            
            # Accuracy
            f.write("## Accuracy Evaluation\n\n")
            accuracy = self.results["accuracy"]
            if "exact_match_score" in accuracy:
                f.write(f"- **Status:** {accuracy['status']}\n")
                f.write(f"- **Exact Match Score:** {accuracy['exact_match_score']:.2%}\n")
                f.write(f"- **Correct Answers:** {accuracy['correct_answers']}/{accuracy['total_queries']}\n\n")
                
                if "by_category" in accuracy:
                    f.write("### By Category\n\n")
                    for category, stats in accuracy["by_category"].items():
                        f.write(f"- **{category.replace('_', ' ').title()}:** {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
                    f.write("\n")
            
            # Explainability
            f.write("## Explainability Evaluation\n\n")
            explainability = self.results["explainability"]
            if "overall_explainability_score" in explainability:
                f.write(f"- **Status:** {explainability['status']}\n")
                f.write(f"- **Overall Score:** {explainability['overall_explainability_score']:.2%}\n")
                f.write(f"- **Reasoning Visibility:** {explainability['reasoning_visibility_rate']:.2%}\n")
                f.write(f"- **Data Attribution:** {explainability['data_attribution_rate']:.2%}\n")
                f.write(f"- **Avg Justification:** {explainability['avg_justification_score']:.2f}/5.0\n\n")
            
            # Performance
            f.write("## Performance Evaluation\n\n")
            performance = self.results["performance"]
            if "text_queries" in performance and "mean" in performance["text_queries"]:
                f.write(f"- **Status:** {performance['overall_status']}\n\n")
                f.write("### Metrics\n\n")
                
                for category, metrics in performance.items():
                    if isinstance(metrics, dict) and "mean" in metrics:
                        f.write(f"**{category.replace('_', ' ').title()}:**\n")
                        f.write(f"- Mean: {metrics['mean']:.3f}s\n")
                        f.write(f"- P95: {metrics['p95']:.3f}s\n")
                        if "target_mean" in metrics:
                            status = "✅" if metrics["meets_mean_target"] else "❌"
                            f.write(f"- Target: {metrics['target_mean']:.3f}s {status}\n")
                        f.write("\n")
            
            # System Info
            f.write("## System Information\n\n")
            for key, value in self.results["system_info"].items():
                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
        
        print(f"📄 Markdown report saved to: {output_path}")


def main():
    """Run comprehensive evaluation suite."""
    import os
    
    # Check for mock mode
    mock_mode = os.getenv("HERMES_TEST_MOCK", "false").lower() == "true"
    
    if mock_mode:
        print("⚠️  Running in MOCK mode (no LLM calls)")
        print("Set HERMES_TEST_MOCK=false to run with actual LLM")
        print("\nNote: Evaluation suite requires actual LLM for meaningful results.")
        return
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_all_evaluations()
    evaluator.save_consolidated_report()
    
    # Exit with appropriate code
    exit_code = 0 if results["overall_status"] in ["PASS", "PARTIAL_PASS"] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
