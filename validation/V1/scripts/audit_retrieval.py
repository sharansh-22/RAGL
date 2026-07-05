"""
RAGL — Validation Study V1: Retrieval Metrics Audit
====================================================
Tests the mathematical correctness of retrieval.py.
"""
import json
import math
import os
import sys
from pathlib import Path

# Add RAGL root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from evaluation.metrics.retrieval import compute_all_retrieval_metrics

def manual_ndcg(retrieved, expected, k=None):
    if k is not None:
        retrieved = retrieved[:k]
    if not retrieved or not expected:
        return 0.0
        
    expected_set = set(expected)
    dcg = sum((1.0 if r in expected_set else 0.0) / math.log2(i + 2) for i, r in enumerate(retrieved))
    
    # ideal
    n_rel = min(len(expected_set), len(retrieved))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0

def run_test(name, retrieved_sources, expected_sources, k=None):
    # Form chunks
    chunks = [{"source": s} for s in retrieved_sources]
    
    # Evaluator output
    eval_metrics = compute_all_retrieval_metrics(chunks, expected_sources, k=k)
    
    # Manual computation
    if k is not None:
        top_k = retrieved_sources[:k]
    else:
        top_k = retrieved_sources
        
    expected_set = set(expected_sources)
    top_k_set = set(top_k)
    hits = top_k_set & expected_set
    
    manual = {}
    manual["hit_rate"] = 1.0 if any(s in expected_set for s in retrieved_sources) else 0.0
    manual["recall_at_k"] = len(hits) / len(expected_set) if expected_set else 0.0
    manual["precision_at_k"] = len([s for s in top_k if s in expected_set]) / len(top_k) if top_k else 0.0
    
    # mrr
    mrr_val = 0.0
    for i, s in enumerate(retrieved_sources):
        if s in expected_set:
            mrr_val = 1.0 / (i + 1)
            break
    manual["mrr"] = mrr_val
    manual["ndcg"] = manual_ndcg(retrieved_sources, expected_sources, k=k)
    
    # Validate
    passed = True
    errors = []
    for metric, expected_val in manual.items():
        eval_val = eval_metrics.get(metric)
        if abs(eval_val - expected_val) > 1e-6:
            passed = False
            errors.append(f"{metric}: manual={expected_val}, eval={eval_val}")
            
    return {
        "test_case": name,
        "retrieved": retrieved_sources,
        "expected": expected_sources,
        "k": k,
        "manual_calculation": manual,
        "evaluator_output": eval_metrics,
        "passed": passed,
        "errors": errors
    }

def main():
    test_cases = [
        ("Perfect Match", ["A", "B", "C"], ["A", "B", "C"], None),
        ("Partial Match Top", ["A", "X", "Y"], ["A", "B"], None),
        ("Late Match", ["X", "Y", "A"], ["A"], None),
        ("No Match", ["X", "Y", "Z"], ["A", "B"], None),
        ("Duplicates in Retrieval", ["A", "A", "B"], ["A", "C"], None),
        ("Empty Expected", ["A"], [], None),
        ("Empty Retrieval", [], ["A"], None),
        ("K Limit 1", ["A", "B", "C"], ["A", "C"], 1),
        ("K Limit 2", ["A", "B", "C"], ["C"], 2),
    ]
    
    results = []
    all_passed = True
    for t in test_cases:
        res = run_test(*t)
        results.append(res)
        if not res["passed"]:
            all_passed = False
            
    out_dir = Path(__file__).resolve().parent.parent
    out_file = out_dir / "retrieval_validation.json"
    with open(out_file, "w") as f:
        json.dump({
            "audit_passed": all_passed,
            "test_cases": results
        }, f, indent=2)
        
    print(f"Retrieval Audit Passed: {all_passed}")
    print(f"Saved evidence to {out_file}")
    
    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
