# Hermes AI - Evaluation Framework

This directory contains the comprehensive evaluation framework for the Hermes AI logistics analytics system.

## Overview

The evaluation framework assesses three core dimensions:

1. **Accuracy** - Correctness and relevance of query responses
2. **Explainability** - Transparency of reasoning and decision-making
3. **Performance** - System responsiveness and efficiency

## Directory Structure

```
evaluation/
├── README.md                      # This file
├── run_all_evaluations.py         # Main evaluation runner
├── test_accuracy.py               # Accuracy tests
├── test_explainability.py         # Explainability tests
├── test_performance.py            # Performance benchmarks
└── results/                       # Output directory (auto-created)
    ├── accuracy_report.json
    ├── explainability_report.json
    ├── performance_metrics.json
    ├── comprehensive_report.json
    └── comprehensive_report.md
```

## Quick Start

### Prerequisites

1. **Data loaded:** Ensure `data/shipments.csv` exists
2. **LLM running:** Local LLM endpoint at `http://localhost:8001/v1` (or configured endpoint)
3. **Dependencies installed:** `pip install -e .` from project root

### Run All Evaluations

```bash
# From project root
cd evaluation

# Run comprehensive evaluation suite
python run_all_evaluations.py
```

This will:
- Run all three evaluation suites
- Generate individual reports for each dimension
- Create a consolidated report (JSON + Markdown)
- Print summary with recommendations

### Run Individual Evaluations

```bash
# Accuracy only
python test_accuracy.py

# Explainability only
python test_explainability.py

# Performance only
python test_performance.py
```

## Evaluation Details

### 1. Accuracy Evaluation (`test_accuracy.py`)

**Purpose:** Verify correctness of query responses against ground truth

**Test Categories:**
- Simple statistics (counts, averages)
- Aggregations (group by, filtering)
- Entity extraction (warehouse/route identification)

**Metrics:**
- **Exact Match (EM):** Percentage of exactly correct answers
- **Numerical Accuracy:** Responses within 5% relative error
- **Category Breakdown:** Accuracy by query type

**Success Criteria:**
- Overall EM ≥ 80%
- Numerical accuracy ≥ 95%

**Example Test:**
```python
Query: "How many shipments were delayed?"
Expected: 234
Actual: 234
Result: ✅ PASS
```

---

### 2. Explainability Evaluation (`test_explainability.py`)

**Purpose:** Assess transparency of system reasoning

**Evaluation Criteria:**

1. **Reasoning Visibility (Binary)**
   - Is classification intent shown?
   - Is confidence score displayed?
   - Are intermediate steps visible?

2. **Data Source Attribution (Binary)**
   - Does response reference specific columns?
   - Are aggregation methods described?
   - Can user trace back to raw data?

3. **Decision Justification (1-5 Scale)**
   - 1: No explanation
   - 2: Minimal ("Based on data analysis")
   - 3: Moderate (mentions method)
   - 4: Detailed (describes logic + sources)
   - 5: Comprehensive (step-by-step with code/logic)

**Metrics:**
- **Reasoning Visibility Rate:** % of queries with visible reasoning
- **Data Attribution Rate:** % of queries with data source references
- **Avg Justification Score:** Mean quality score (1-5)
- **Overall Explainability Score:** Weighted combination

**Success Criteria:**
- Reasoning Visibility: 100%
- Data Attribution: ≥ 80%
- Avg Justification: ≥ 3.5/5
- Overall Score: ≥ 0.75

**Example Evaluation:**
```
Query: "Which warehouse has most delays?"

Reasoning Chain:
✅ Classification: statistics (0.91 confidence)
✅ Data Source: Analyzing 'warehouse' and 'delay_minutes' columns
✅ Method: Grouping by warehouse, counting delays > 0
✅ Result: WH_02 with 87 delayed shipments

Scores:
- Reasoning Visibility: ✅ Yes
- Data Attribution: ✅ Yes
- Justification: 4/5 (Detailed)
```

---

### 3. Performance Evaluation (`test_performance.py`)

**Purpose:** Measure system responsiveness and efficiency

**Benchmarks:**

| Operation | Target (Mean) | Target (P95) |
|-----------|---------------|--------------|
| Data Loading | < 3 sec | < 5 sec |
| Classification | < 1 sec | < 2 sec |
| Text Query | < 5 sec | < 8 sec |
| Chart Query | < 8 sec | < 12 sec |

**Metrics:**
- Mean response time
- Median response time
- P95 (95th percentile)
- P99 (99th percentile)
- Min/Max times

**Success Criteria:**
- 90% of queries meet mean target
- 95% of queries meet P95 target
- No query exceeds 15 seconds

**Example Output:**
```
Text Query Performance:
  Mean: 3.5s ✅ (target: 5.0s)
  P95: 6.3s ✅ (target: 8.0s)
  Status: PASS
```

---

## Evaluation Reports

### JSON Report Structure

```json
{
  "evaluation_date": "2025-11-09T14:30:00Z",
  "test_suite_version": "1.0.0",
  "system_info": {
    "python_version": "3.11.5",
    "llm_model": "qwen2.5:latest",
    "llm_endpoint": "http://localhost:8001/v1"
  },
  "accuracy": {
    "total_queries": 15,
    "correct_answers": 13,
    "exact_match_score": 0.867,
    "status": "PASS"
  },
  "explainability": {
    "reasoning_visibility_rate": 1.0,
    "data_attribution_rate": 0.86,
    "avg_justification_score": 3.8,
    "overall_explainability_score": 0.82,
    "status": "PASS"
  },
  "performance": {
    "text_queries": {
      "mean": 3.5,
      "p95": 6.3,
      "meets_mean_target": true
    },
    "overall_status": "PASS"
  },
  "overall_status": "PASS",
  "summary": "Accuracy: 86.7% (13/15 queries correct) | Explainability: 82.0% (avg justification: 3.8/5) | Performance: 3.50s avg response time"
}
```

### Markdown Report

The system also generates a human-readable markdown report (`comprehensive_report.md`) with:
- Executive summary
- Detailed metrics by category
- Pass/fail indicators
- System configuration snapshot

---

## Usage Scenarios

### Scenario 1: Pre-Deployment Testing

Before deploying Hermes to production:

```bash
# Run full evaluation
python run_all_evaluations.py

# Review consolidated report
cat results/comprehensive_report.md

# Check if all tests pass
echo $?  # Should be 0 for PASS
```

### Scenario 2: Regression Testing

After making code changes:

```bash
# Run evaluations
python run_all_evaluations.py

# Compare with baseline
diff results/comprehensive_report.json baseline_report.json
```

### Scenario 3: Performance Profiling

To identify bottlenecks:

```bash
# Run performance tests only
python test_performance.py

# Review detailed metrics
cat results/performance_metrics.json | jq
```

### Scenario 4: Continuous Integration

In CI pipeline (`.github/workflows/evaluate.yml`):

```yaml
- name: Run Evaluation Suite
  run: |
    cd evaluation
    python run_all_evaluations.py
  env:
    HERMES_TEST_MOCK: false  # Use actual LLM
```

---

## Customization

### Adding Custom Tests

**Example: Add a new accuracy test**

Edit `test_accuracy.py`:

```python
def test_custom_category(self) -> Dict[str, Any]:
    """Test custom query category."""
    tests = [
        {
            "query": "Your custom query",
            "expected": 42,
            "type": "number"
        }
    ]
    
    # ... evaluation logic ...
```

### Adjusting Performance Targets

Edit targets in `test_performance.py`:

```python
self.targets = {
    "data_loading": {"mean": 2.0, "p95": 4.0},  # Stricter targets
    "text_query": {"mean": 3.0, "p95": 5.0}
}
```

### Modifying Success Criteria

Edit success thresholds in individual test files:

```python
# test_accuracy.py
assert overall_accuracy >= 0.90, "Increased accuracy requirement"

# test_explainability.py
assert explainability_score >= 0.80, "Higher explainability bar"
```

---

## Troubleshooting

### LLM Endpoint Unreachable

**Error:** `Connection refused` or `Timeout`

**Solution:**
```bash
# Check if LLM server is running
curl http://localhost:8001/v1/models

# Start LLM server (example with Ollama)
ollama serve
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'hermes'`

**Solution:**
```bash
# Install package in editable mode
cd /path/to/Hermes
pip install -e .
```

### Tests Timing Out

**Error:** Tests hang or take very long

**Possible Causes:**
- LLM server overloaded
- Network latency
- Large dataset

**Solution:**
- Reduce test iterations
- Use smaller test dataset
- Increase timeout limits

### Mock Mode for CI

To run tests without actual LLM (dry-run):

```bash
export HERMES_TEST_MOCK=true
python run_all_evaluations.py
```

**Note:** Mock mode skips actual evaluations and returns placeholder results.

---

## Interpretation Guide

### What Do the Scores Mean?

**Accuracy Score:**
- **> 90%:** Excellent - Production ready
- **80-90%:** Good - Minor improvements needed
- **70-80%:** Fair - Review failed queries, improve prompts
- **< 70%:** Poor - Significant LLM tuning required

**Explainability Score:**
- **> 0.85:** Excellent - Users can fully understand decisions
- **0.75-0.85:** Good - Most reasoning is transparent
- **0.65-0.75:** Fair - Add more reasoning display
- **< 0.65:** Poor - Insufficient transparency

**Performance (Mean Response Time):**
- **< 3s:** Excellent - Very responsive
- **3-5s:** Good - Acceptable for most use cases
- **5-8s:** Fair - Consider optimization
- **> 8s:** Poor - Optimize LLM or add caching

### When to Re-evaluate

**Mandatory:**
- After major code changes
- Before production deployment
- After LLM model updates

**Recommended:**
- Weekly during active development
- Monthly in production
- After configuration changes

---

## Best Practices

1. **Baseline First:** Run evaluation on clean main branch to establish baseline
2. **Version Control Reports:** Commit evaluation reports to track changes over time
3. **Automate:** Integrate into CI/CD pipeline
4. **Monitor Trends:** Track metrics over time, not just pass/fail
5. **Investigate Failures:** Always review failed test cases for patterns
6. **Document Changes:** Note what changed when metrics improve/degrade

---

## Support

**Issues with evaluation framework?**
- Check `logs/hermes.log` for detailed error messages
- Ensure LLM endpoint is reachable and responsive
- Verify data files exist and are properly formatted

**Questions about metrics?**
- See `DELIVERABLES.md` Section 3 for detailed methodology
- Review individual test file docstrings

**Want to contribute?**
- Add new test cases to existing evaluators
- Create new evaluation dimensions (e.g., security, scalability)
- Improve reporting formats

---

## Next Steps

After running evaluations:

1. **Review Reports:** Check `results/comprehensive_report.md`
2. **Address Failures:** Fix any queries that failed accuracy tests
3. **Optimize Performance:** If response times exceed targets, profile bottlenecks
4. **Enhance Explainability:** Improve reasoning display if scores are low
5. **Document:** Update project README with evaluation results

---

**Last Updated:** November 9, 2025  
**Framework Version:** 1.0.0  
**Maintainer:** Phi Tran
