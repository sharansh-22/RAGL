"""
RAGL — Experiment Comparison Tool
=================================
Loads cached benchmark JSONs from multiple experiments, aggregates metrics,
computes deltas against a baseline, and generates a comparative report and plots.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAGL_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def load_experiment_metrics(experiment_dir: Path) -> dict:
    """Loads metrics from an experiment's cache directory."""
    if not experiment_dir.exists():
        logger.warning(f"Cache directory not found: {experiment_dir}")
        return None
        
    summary_path = experiment_dir / "summary.json"
    if not summary_path.exists():
        logger.warning(f"No summary.json in {experiment_dir}")
        return None
        
    with open(summary_path) as f:
        summary = json.load(f)
        
    results_files = summary.get("results_files", [])
    
    retrieval_data = {}
    generation_data = {}
    latency_data = {"retrieval_ms": [], "generation_ms": [], "total_ms": []}
    
    for filename in results_files:
        filepath = experiment_dir / filename
        if not filepath.exists():
            continue
        with open(filepath) as f:
            data = json.load(f)
            
            for k, v in data.get("retrieval_metrics", {}).items():
                retrieval_data.setdefault(k, []).append(v)
            for k, v in data.get("generation_metrics", {}).items():
                generation_data.setdefault(k, []).append(v)
            if "latency_ms" in data:
                latency_data["total_ms"].append(data["latency_ms"])
                
    # Calculate means
    means = {
        "retrieval": {k: np.mean(v) for k, v in retrieval_data.items()},
        "generation": {k: np.mean(v) for k, v in generation_data.items()},
        "latency": {k: np.mean(v) for k, v in latency_data.items() if v}
    }
    
    return {
        "config": summary.get("config", {}),
        "means": means
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline experiment name (e.g. A0)")
    parser.add_argument("--candidates", required=True, help="Candidate experiment group (e.g. A1)")
    args = parser.parse_args()
    
    baseline_dir = RAGL_ROOT / "evaluation" / "cache" / args.baseline
    candidates_dir = RAGL_ROOT / "evaluation" / "cache" / args.candidates
    
    baseline_data = load_experiment_metrics(baseline_dir)
    if not baseline_data:
        logger.error("Baseline data could not be loaded. Exiting.")
        return
        
    candidates = {}
    if candidates_dir.exists():
        for d in sorted(candidates_dir.iterdir()):
            if d.is_dir():
                data = load_experiment_metrics(d)
                if data:
                    candidates[d.name] = data
                    
    logger.info(f"Loaded baseline {args.baseline} and {len(candidates)} candidates.")
    
    # -----------------------------------------------------------------------
    # Generate Output JSON
    # -----------------------------------------------------------------------
    out_dir = RAGL_ROOT / "evaluation" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "baseline": args.baseline,
        "baseline_metrics": baseline_data["means"],
        "candidates": {name: data["means"] for name, data in candidates.items()}
    }
    with open(out_dir / f"{args.candidates}_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # -----------------------------------------------------------------------
    # Generate Plots
    # -----------------------------------------------------------------------
    plots_dir = RAGL_ROOT / "evaluation" / "plots" / args.candidates
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    model_names = [args.baseline] + list(candidates.keys())
    
    # Retrieval Plot
    retrieval_metrics = list(baseline_data["means"]["retrieval"].keys())
    if retrieval_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(retrieval_metrics))
        width = 0.8 / len(model_names)
        
        for i, name in enumerate(model_names):
            data = baseline_data if name == args.baseline else candidates[name]
            means = [data["means"]["retrieval"].get(m, 0) for m in retrieval_metrics]
            ax.bar(x + i*width - 0.4 + width/2, means, width, label=name)
            
        ax.set_ylabel('Score')
        ax.set_title(f'Retrieval Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(retrieval_metrics)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(plots_dir / "retrieval_comparison.png")
        plt.close()
        
    # Generation Plot
    generation_metrics = list(baseline_data["means"]["generation"].keys())
    if generation_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(generation_metrics))
        width = 0.8 / len(model_names)
        
        for i, name in enumerate(model_names):
            data = baseline_data if name == args.baseline else candidates[name]
            means = [data["means"]["generation"].get(m, 0) for m in generation_metrics]
            ax.bar(x + i*width - 0.4 + width/2, means, width, label=name)
            
        ax.set_ylabel('Score')
        ax.set_title(f'Generation Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(generation_metrics)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(plots_dir / "generation_comparison.png")
        plt.close()
        
    # Latency Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    latencies = []
    for name in model_names:
        data = baseline_data if name == args.baseline else candidates[name]
        latencies.append(data["means"]["latency"].get("total_ms", 0))
    ax.bar(model_names, latencies, color='#DD8452')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Total Latency Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "latency_comparison.png")
    plt.close()
    
    # -----------------------------------------------------------------------
    # Generate Report
    # -----------------------------------------------------------------------
    reports_dir = RAGL_ROOT / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{args.candidates}_report.md"
    
    lines = [
        f"# RAGL {args.candidates} — Embedding Model Study",
        f"",
        f"**Objective**: Determine whether changing only the embedding model produces a measurable improvement in retrieval and generation quality.",
        f"**Baseline**: {args.baseline}",
        f"**Candidates**: {', '.join(candidates.keys())}",
        f"",
        f"## Constraints",
        f"- Chunk size and overlap remained identical to {args.baseline}.",
        f"- The LLM (`llama3:8b`) remained identical.",
        f"- The FAISS index parameters remained identical.",
        f"- Comparison uses cached benchmark JSONs. No previous experiments were regenerated.",
        f"",
        f"## Benchmark Comparison Table",
        f""
    ]
    
    all_metrics = ["hit_rate", "mrr", "ndcg", "recall_at_k", "precision_at_k", "faithfulness", "groundedness", "relevancy"]
    
    header = "| Model | " + " | ".join(all_metrics) + " | Latency (ms) |"
    lines.append(header)
    lines.append("|-------|" + "|".join(["---"] * len(all_metrics)) + "|--------------|")
    
    for name in model_names:
        data = baseline_data if name == args.baseline else candidates[name]
        row = [name]
        for m in all_metrics:
            if m in data["means"]["retrieval"]:
                row.append(f"{data['means']['retrieval'][m]:.4f}")
            elif m in data["means"]["generation"]:
                row.append(f"{data['means']['generation'][m]:.4f}")
            else:
                row.append("N/A")
        lat = data["means"]["latency"].get("total_ms", 0)
        row.append(f"{lat:.1f}")
        lines.append("| " + " | ".join(row) + " |")
        
    lines.extend([
        f"",
        f"## Engineering Recommendations",
        f"",
        f"> **PENDING**: Human Engineering Review will be appended here after qualitative inspection."
    ])
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
        
    logger.info(f"Comparison complete. Wrote {report_path}")

if __name__ == "__main__":
    main()
