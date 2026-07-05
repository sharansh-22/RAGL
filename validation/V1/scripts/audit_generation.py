"""
RAGL — Validation Study V1: Generation Metrics Audit
====================================================
Tests Faithfulness, Groundedness, and Relevancy models on synthetic cases.
"""
import json
import sys
from pathlib import Path

# Add RAGL root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from evaluation.metrics.generation import faithfulness, groundedness, relevancy

def run_tests():
    results = {}
    
    # -----------------------------------------------------------------
    # 4. Faithfulness Audit & 7. Sensitivity Analysis
    # -----------------------------------------------------------------
    context = "Paris is the capital of France. The Eiffel Tower is in Paris. The Louvre is a famous museum."
    
    # Test 1: Perfect
    a_perfect = "Paris is the capital of France and the Eiffel Tower is there."
    f_perf = faithfulness(a_perfect, context)
    g_perf = groundedness(a_perfect, context)
    
    # Test 2: One Hallucination
    a_hal1 = "Paris is the capital of France and the Eiffel Tower is there. Berlin is the capital of Germany."
    f_hal1 = faithfulness(a_hal1, context)
    g_hal1 = groundedness(a_hal1, context)
    
    # Test 3: Two Hallucinations
    a_hal2 = "Paris is the capital of France and the Eiffel Tower is there. Berlin is the capital of Germany. Madrid is the capital of Spain."
    f_hal2 = faithfulness(a_hal2, context)
    
    # Test 4: Completely Irrelevant
    a_irrel = "The sun is a star. Water boils at 100 degrees."
    f_irrel = faithfulness(a_irrel, context)
    
    # -----------------------------------------------------------------
    # 5. Groundedness Audit (Sentence-level against chunks)
    # -----------------------------------------------------------------
    chunks = [
        "The first step is to initialize the weights.",
        "The second step is the forward pass.",
        "The third step is calculating the loss."
    ]
    # Sentence 1 grounded in chunk 0. Sentence 2 grounded in chunk 1.
    a_ground = "Weights are initialized first. Then you do a forward pass."
    g_multi_chunk = groundedness(a_ground, chunks)
    
    # Only 1 sentence grounded out of 2
    a_partial = "Weights are initialized first. Then you eat a sandwich."
    g_partial_chunk = groundedness(a_partial, chunks)
    
    # -----------------------------------------------------------------
    # 6. Relevancy Audit
    # -----------------------------------------------------------------
    q = "Explain gradient descent."
    a_rel = "Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function."
    a_unrel = "I like to eat apples and bananas."
    
    r_rel = relevancy(a_rel, q)
    r_unrel = relevancy(a_unrel, q)
    
    # Record everything
    results["faithfulness_sensitivity"] = {
        "perfect": f_perf,
        "one_hallucination": f_hal1,
        "two_hallucinations": f_hal2,
        "completely_irrelevant": f_irrel
    }
    
    results["groundedness_sensitivity"] = {
        "perfect": g_perf,
        "one_hallucination": g_hal1,
        "multi_chunk_perfect": g_multi_chunk,
        "multi_chunk_partial": g_partial_chunk
    }
    
    results["relevancy"] = {
        "highly_relevant": r_rel,
        "highly_irrelevant": r_unrel
    }
    
    # -----------------------------------------------------------------
    # Validation Logic
    # -----------------------------------------------------------------
    passed = True
    errors = []
    
    if f_perf < 0.9: errors.append(f"Faithfulness (perfect) too low: {f_perf}")
    if f_hal1 >= f_perf: errors.append("Faithfulness did not drop after 1 hallucination")
    if f_hal2 >= f_hal1: errors.append("Faithfulness did not drop after 2 hallucinations")
    if f_irrel > 0.1: errors.append("Faithfulness (irrelevant) too high")
    
    if g_multi_chunk < 0.9: errors.append(f"Groundedness (multi chunk perfect) too low: {g_multi_chunk}")
    if g_partial_chunk > 0.6 or g_partial_chunk < 0.4: errors.append(f"Groundedness (partial) expected ~0.5, got {g_partial_chunk}")
    
    if r_rel < 0.8: errors.append(f"Relevancy (relevant) too low: {r_rel}")
    if r_unrel > 0.3: errors.append(f"Relevancy (irrelevant) too high: {r_unrel}")
    
    if errors:
        passed = False
        
    out_dir = Path(__file__).resolve().parent.parent
    out_file = out_dir / "metric_validation.json"
    
    with open(out_file, "w") as f:
        json.dump({
            "audit_passed": passed,
            "errors": errors,
            "results": results
        }, f, indent=2)
        
    print(f"Generation Audit Passed: {passed}")
    for err in errors:
        print(f" - Error: {err}")
    print(f"Saved evidence to {out_file}")

    # Generate a plot for sensitivity
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = ["Perfect", "1 Hallucination", "2 Hallucinations", "Irrelevant"]
    y = [f_perf, f_hal1, f_hal2, f_irrel]
    ax.plot(x, y, marker='o', linestyle='-', color='blue')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Faithfulness Score")
    ax.set_title("Faithfulness Sensitivity to Hallucinations")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "faithfulness_sensitivity.png")
    plt.close()

if __name__ == "__main__":
    run_tests()
