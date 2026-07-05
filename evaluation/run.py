"""
RAGL — Evaluation Runner
==========================
Reusable benchmark runner for any RAGL experiment.

Usage:
    python evaluation/run.py --experiment A0
    python evaluation/run.py --experiment A0 --datasets easy medium
    python evaluation/run.py --experiment A0 --datasets all

This runner:
  1. Loads the experiment config (A/A0/config.py, A/A1/config.py, ...)
  2. Builds or loads the FAISS index
  3. Loads evaluation datasets
  4. Runs every query through the RAG pipeline
  5. Computes retrieval + generation + performance metrics
  6. Saves reports, plots, and cached outputs
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

# Ensure RAGL root is on the path
RAGL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAGL_ROOT))

from core.chunker import load_documents, chunk_documents
from core.embedder import Embedder
from core.indexer import build_index, save_index, load_index
from core.rag import RAGPipeline
from evaluation.metrics.retrieval import compute_all_retrieval_metrics
from evaluation.metrics.generation import compute_all_generation_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_datasets(
    datasets_dir: str,
    dataset_names: list[str] | None = None,
) -> list[dict]:
    """
    Load evaluation datasets from JSON files.

    Args:
        datasets_dir: Path to evaluation/datasets/.
        dataset_names: List of dataset names (without .json) to load.
                       If None or ["all"], loads all .json files.

    Returns:
        Combined list of query dicts.
    """
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    if dataset_names is None or "all" in dataset_names:
        json_files = sorted(datasets_path.glob("*.json"))
    else:
        json_files = []
        for name in dataset_names:
            f = datasets_path / f"{name}.json"
            if f.exists():
                json_files.append(f)
            else:
                logger.warning(f"Dataset not found: {f}")

    if not json_files:
        raise FileNotFoundError(f"No datasets found in: {datasets_dir}")

    all_queries = []
    for f in json_files:
        with open(f) as fp:
            queries = json.load(fp)
            logger.info(f"Loaded {len(queries)} queries from {f.name}")
            all_queries.extend(queries)

    logger.info(f"Total evaluation queries: {len(all_queries)}")
    return all_queries


# ---------------------------------------------------------------------------
# Experiment Config Loading
# ---------------------------------------------------------------------------

def load_experiment_config(experiment_name: str) -> dict:
    """
    Dynamically load config from A/<experiment>/config.py.

    Returns a dict of all uppercase attributes from the config module.
    """
    import importlib
    module_name = f"A.{experiment_name}.config"

    try:
        config_module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            f"Expected module: {module_name}"
        )

    config = {}
    for key in dir(config_module):
        if key.isupper():
            config[key] = getattr(config_module, key)

    return config


# ---------------------------------------------------------------------------
# Index Building / Loading
# ---------------------------------------------------------------------------

def build_or_load_index(config: dict, force_rebuild: bool = False):
    """
    Build the FAISS index (and chunks) or load from cache.

    Returns:
        (embedder, index, chunks)
    """
    index_dir = Path(config["INDEX_DIR"])
    index_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.json"

    embedder = Embedder(model_name=config["EMBEDDING_MODEL"])
    chunker_strategy = config.get("CHUNKER_STRATEGY", "sentence")

    if index_path.exists() and chunks_path.exists() and not force_rebuild:
        logger.info("Loading existing index from cache...")
        index = load_index(str(index_path))
        with open(chunks_path) as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks, {index.ntotal} vectors")
        return embedder, index, chunks

    logger.info(f"Building index from documents using strategy: {chunker_strategy}")
    documents = load_documents(config["DATA_DIR"])
    
    chunk_start = time.perf_counter()
    chunks = chunk_documents(
        documents,
        chunk_size=config["CHUNK_SIZE"],
        chunk_overlap=config["CHUNK_OVERLAP"],
        strategy=chunker_strategy
    )
    chunk_time = time.perf_counter() - chunk_start

    # Embed all chunks
    logger.info(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embed_start = time.perf_counter()
    embeddings = embedder.embed(texts)
    embed_time = time.perf_counter() - embed_start

    # Build FAISS index
    index_start = time.perf_counter()
    index = build_index(embeddings)
    index_time = time.perf_counter() - index_start

    # Save to disk
    index_dir.mkdir(parents=True, exist_ok=True)
    save_index(index, str(index_path))
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)
        
    index_size_bytes = os.path.getsize(index_path)

    # Calculate and save chunk statistics
    if chunks:
        chunk_sizes = [len(c["text"]) for c in chunks]
        # Approximate overlap (just comparing length of (text_i, text_i-1) isn't exact,
        # but we can record the configured overlap for now).
        stats = {
            "num_chunks": int(len(chunks)),
            "avg_size": float(np.mean(chunk_sizes)),
            "med_size": float(np.median(chunk_sizes)),
            "max_size": float(np.max(chunk_sizes)),
            "min_size": float(np.min(chunk_sizes)),
            "configured_overlap": int(config["CHUNK_OVERLAP"]),
            "embedding_time_sec": float(embed_time),
            "index_build_time_sec": float(index_time),
            "chunking_time_sec": float(chunk_time),
            "index_size_bytes": int(index_size_bytes),
        }
        with open(index_dir / "chunk_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
            
        # Save sample chunks
        sample_chunks_path = index_dir / "sample_chunks.md"
        with open(sample_chunks_path, "w") as f:
            f.write("# Representative Chunks\\n\\n")
            for i, c in enumerate(chunks[:3]):
                f.write(f"## Chunk {i}\\n")
                f.write(f"**Source**: {c['source']}\\n\\n")
                f.write(f"```text\\n{c['text']}\\n```\\n\\n")

    logger.info(f"Saved index and {len(chunks)} chunks to {index_dir}")

    return embedder, index, chunks


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    experiment_name: str,
    results: list[dict],
    config: dict,
    output_dir: str,
) -> str:
    """Generate a markdown report from benchmark results."""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{experiment_name}_report.md"

    # Aggregate metrics
    retrieval_metrics = {}
    generation_metrics = {}
    latencies = {"retrieval_ms": [], "generation_ms": [], "total_ms": []}

    for r in results:
        for key, val in r.get("retrieval_metrics", {}).items():
            retrieval_metrics.setdefault(key, []).append(val)
        for key, val in r.get("generation_metrics", {}).items():
            generation_metrics.setdefault(key, []).append(val)
        for key in latencies:
            if key in r.get("latency", {}):
                latencies[key].append(r["latency"][key])

    lines = [
        f"# {experiment_name} — Benchmark Report",
        f"",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Experiment**: {experiment_name} (Reference Implementation)",
        f"**Queries evaluated**: {len(results)}",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Embedding Model | `{config.get('EMBEDDING_MODEL', 'N/A')}` |",
        f"| LLM Model | `{config.get('LLM_MODEL', 'N/A')}` |",
        f"| Chunk Size | {config.get('CHUNK_SIZE', 'N/A')} |",
        f"| Chunk Overlap | {config.get('CHUNK_OVERLAP', 'N/A')} |",
        f"| Top-K | {config.get('TOP_K', 'N/A')} |",
        f"",
        f"## Retrieval Metrics",
        f"",
        f"| Metric | Mean | Std | Min | Max |",
        f"|--------|------|-----|-----|-----|",
    ]

    for key, values in sorted(retrieval_metrics.items()):
        arr = np.array(values)
        lines.append(
            f"| {key} | {arr.mean():.4f} | {arr.std():.4f} | "
            f"{arr.min():.4f} | {arr.max():.4f} |"
        )

    lines.extend([
        f"",
        f"## Generation Metrics",
        f"",
        f"| Metric | Mean | Std | Min | Max |",
        f"|--------|------|-----|-----|-----|",
    ])

    for key, values in sorted(generation_metrics.items()):
        arr = np.array(values)
        lines.append(
            f"| {key} | {arr.mean():.4f} | {arr.std():.4f} | "
            f"{arr.min():.4f} | {arr.max():.4f} |"
        )

    lines.extend([
        f"",
        f"## Performance",
        f"",
        f"| Stage | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |",
        f"|-------|-----------|----------|----------|----------|",
    ])

    for key in ["retrieval_ms", "generation_ms", "total_ms"]:
        if latencies[key]:
            arr = np.array(latencies[key])
            lines.append(
                f"| {key} | {arr.mean():.1f} | {arr.std():.1f} | "
                f"{arr.min():.1f} | {arr.max():.1f} |"
            )

    lines.extend([
        f"",
        f"## Per-Query Results",
        f"",
    ])

    for r in results:
        lines.append(f"### {r['id']}")
        lines.append(f"**Query**: {r['query']}")
        lines.append(f"**Difficulty**: {r.get('difficulty', 'N/A')}")
        lines.append(f"**Category**: {r.get('category', 'N/A')}")
        lines.append(f"")

        ret_m = r.get("retrieval_metrics", {})
        lines.append(f"Retrieval: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(ret_m.items())
        ))

        gen_m = r.get("generation_metrics", {})
        lines.append(f"Generation: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(gen_m.items())
        ))

        lat = r.get("latency", {})
        lines.append(f"Latency: " + ", ".join(
            f"{k}={v:.1f}ms" for k, v in lat.items()
        ))
        lines.append(f"")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"Report saved to: {report_path}")
    return str(report_path)


# ---------------------------------------------------------------------------
# Plot Generation
# ---------------------------------------------------------------------------

def generate_plots(
    experiment_name: str,
    results: list[dict],
    output_dir: str,
) -> None:
    """Generate evaluation metric plots."""
    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Collect all metrics
    retrieval_data = {}
    generation_data = {}
    latency_data = {"retrieval_ms": [], "generation_ms": [], "total_ms": []}

    for r in results:
        for key, val in r.get("retrieval_metrics", {}).items():
            retrieval_data.setdefault(key, []).append(val)
        for key, val in r.get("generation_metrics", {}).items():
            generation_data.setdefault(key, []).append(val)
        for key in latency_data:
            if key in r.get("latency", {}):
                latency_data[key].append(r["latency"][key])

    # Plot retrieval metrics
    if retrieval_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = sorted(retrieval_data.keys())
        means = [np.mean(retrieval_data[m]) for m in metrics]
        stds = [np.std(retrieval_data[m]) for m in metrics]
        bars = ax.bar(metrics, means, yerr=stds, capsize=5,
                       color="#4C72B0", edgecolor="black", alpha=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"{experiment_name} — Retrieval Metrics")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "retrieval_metrics.png", dpi=150)
        plt.close()

    # Plot generation metrics
    if generation_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = sorted(generation_data.keys())
        means = [np.mean(generation_data[m]) for m in metrics]
        stds = [np.std(generation_data[m]) for m in metrics]
        bars = ax.bar(metrics, means, yerr=stds, capsize=5,
                       color="#55A868", edgecolor="black", alpha=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"{experiment_name} — Generation Metrics")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "generation_metrics.png", dpi=150)
        plt.close()

    # Plot latency distribution
    if any(latency_data.values()):
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = []
        values = []
        for key in ["retrieval_ms", "generation_ms", "total_ms"]:
            if latency_data[key]:
                labels.append(key.replace("_ms", ""))
                values.append(latency_data[key])

        ax.boxplot(values, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="#DD8452", alpha=0.7))
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{experiment_name} — Latency Distribution")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "latency.png", dpi=150)
        plt.close()

    logger.info(f"Plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# Cache Results
# ---------------------------------------------------------------------------

def cache_results(
    experiment_name: str,
    results: list[dict],
    config: dict,
    output_dir: str,
) -> None:
    """Save per-query cached results as immutable experiment records."""
    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save individual query results (Level 2 Evidence)
    for r in results:
        query_file = cache_dir / f"{r['id']}.json"
        
        # Build the exact schema requested
        level2_evidence = {
            "experiment": experiment_name,
            "configuration": {
                "embedding_model": config.get("EMBEDDING_MODEL"),
                "llm": config.get("LLM_MODEL"),
                "chunk_size": config.get("CHUNK_SIZE"),
                "chunk_overlap": config.get("CHUNK_OVERLAP"),
                "top_k": config.get("TOP_K")
            },
            "query_id": r["id"],
            "query": r["query"],
            "prompt": r["prompt"],
            "answer": r["answer"],
            "retrieved_documents": r["retrieved_sources"],
            "retrieved_chunk_ids": [c.get("id", i) for i, c in enumerate(r["retrieved_chunks"])],
            "retrieval_metrics": r["retrieval_metrics"],
            "generation_metrics": r["generation_metrics"],
            "latency_ms": r["latency"].get("total_ms")
        }
        
        with open(query_file, "w") as f:
            json.dump(level2_evidence, f, indent=2, default=str)

    # Save aggregate summary
    summary = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config": {k: str(v) for k, v in config.items()},
        "num_queries": len(results),
        "results_files": [f"{r['id']}.json" for r in results],
    }
    with open(cache_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save retrieval visualization
    viz_path = cache_dir / "retrieval_visualization.md"
    with open(viz_path, "w") as f:
        f.write(f"# Retrieval Visualization - {experiment_name}\\n\\n")
        for r in results:
            f.write(f"## Query: {r['query']}\\n\\n")
            f.write(f"**Query ID**: {r['id']}\\n\\n")
            for chunk in r['retrieved_chunks']:
                f.write(f"### Score: {chunk.get('score', 0):.4f} (Rank {chunk.get('rank', 0)})\\n")
                f.write(f"**Source**: {chunk['source']}\\n\\n")
                f.write(f"```text\\n{chunk['text']}\\n```\\n\\n")
            f.write("---\\n\\n")

    logger.info(f"Cached {len(results)} results to: {cache_dir}")


# ---------------------------------------------------------------------------
# Main Benchmark Loop
# ---------------------------------------------------------------------------

def run_benchmark(
    experiment_name: str,
    dataset_names: list[str] | None = None,
    force_rebuild: bool = False,
) -> list[dict]:
    """
    Run the full benchmark for an experiment.

    Args:
        experiment_name: e.g., "A0", "A1", "A2"
        dataset_names: Which datasets to load (None = all)
        force_rebuild: If True, rebuild the FAISS index from scratch

    Returns:
        List of per-query result dicts.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    console = Console()

    console.print(f"\n[bold blue]RAGL Benchmark Runner[/bold blue]")
    console.print(f"Experiment: [bold]{experiment_name}[/bold]")
    console.print()

    # Load experiment config
    config = load_experiment_config(experiment_name)
    console.print(f"[green]✓[/green] Config loaded: {experiment_name}")

    # Build or load index
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Building/loading index...", total=None)
        embedder, index, chunks = build_or_load_index(config, force_rebuild)
        progress.update(task, completed=True)
    console.print(
        f"[green]✓[/green] Index ready: {index.ntotal} vectors, "
        f"{len(chunks)} chunks"
    )

    # Load datasets
    queries = load_datasets(config["EVAL_DATASETS_DIR"], dataset_names)
    console.print(
        f"[green]✓[/green] Loaded {len(queries)} evaluation queries"
    )

    # Create pipeline
    pipeline = RAGPipeline(
        embedder=embedder,
        index=index,
        chunks=chunks,
        model_name=config["LLM_MODEL"],
        top_k=config["TOP_K"],
    )

    # Run queries
    results = []
    console.print(f"\n[bold]Running {len(queries)} queries...[/bold]\n")

    for i, q in enumerate(queries, start=1):
        console.print(
            f"  [{i}/{len(queries)}] {q['id']}: {q['query'][:60]}..."
        )

        # Run RAG pipeline
        result = pipeline.query(q["query"])

        # Compute retrieval metrics
        ret_metrics = compute_all_retrieval_metrics(
            result["retrieved_chunks"],
            q.get("expected_sources", []),
            k=config["TOP_K"],
        )
        
        # Calculate average context length
        if result["retrieved_chunks"]:
            avg_context_len = np.mean([len(c["text"]) for c in result["retrieved_chunks"]])
        else:
            avg_context_len = 0.0
        ret_metrics["avg_retrieved_context_len"] = float(avg_context_len)

        # Compute generation metrics
        gen_metrics = compute_all_generation_metrics(
            answer=result["answer"],
            query=q["query"],
            context=result["retrieved_chunks"],
            nli_model_name=config.get("NLI_MODEL", "cross-encoder/nli-deberta-v3-small"),
            ce_model_name=config.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        )

        # Assemble full result
        full_result = {
            "id": q["id"],
            "query": q["query"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "expected_sources": q.get("expected_sources", []),
            "retrieved_chunks": result["retrieved_chunks"],
            "retrieved_sources": list(set(
                c["source"] for c in result["retrieved_chunks"]
            )),
            "prompt": result["prompt"],
            "answer": result["answer"],
            "latency": result["latency"],
            "retrieval_metrics": ret_metrics,
            "generation_metrics": gen_metrics,
        }
        results.append(full_result)

        # Print brief status
        console.print(
            f"         hit={ret_metrics['hit_rate']:.0f} "
            f"recall={ret_metrics['recall_at_k']:.2f} "
            f"faith={gen_metrics['faithfulness']:.2f} "
            f"relev={gen_metrics['relevancy']:.2f} "
            f"latency={result['latency']['total_ms']:.0f}ms"
        )

    # Generate outputs
    console.print(f"\n[bold]Generating outputs...[/bold]")

    report_path = generate_report(
        experiment_name, results, config, config["EVAL_REPORTS_DIR"]
    )
    console.print(f"[green]✓[/green] Report: {report_path}")

    generate_plots(experiment_name, results, config["EVAL_PLOTS_DIR"])
    console.print(f"[green]✓[/green] Plots: {config['EVAL_PLOTS_DIR']}")

    cache_results(experiment_name, results, config, config["EVAL_CACHE_DIR"])
    console.print(f"[green]✓[/green] Cache: {config['EVAL_CACHE_DIR']}")

    # Summary table
    console.print()
    table = Table(title=f"{experiment_name} — Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")

    all_metrics = {}
    for r in results:
        for k, v in r.get("retrieval_metrics", {}).items():
            all_metrics.setdefault(k, []).append(v)
        for k, v in r.get("generation_metrics", {}).items():
            all_metrics.setdefault(k, []).append(v)

    for key in sorted(all_metrics.keys()):
        mean = np.mean(all_metrics[key])
        table.add_row(key, f"{mean:.4f}")

    latencies_total = [r["latency"]["total_ms"] for r in results]
    table.add_row("avg_latency_ms", f"{np.mean(latencies_total):.1f}")

    console.print(table)
    console.print(f"\n[bold green]Benchmark complete.[/bold green]\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAGL Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/run.py --experiment A0
  python evaluation/run.py --experiment A0 --datasets easy medium
  python evaluation/run.py --experiment A0 --datasets all --rebuild
        """,
    )
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        help="Experiment name (e.g., A0, A1, A2)",
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="*",
        default=None,
        help="Dataset names to load (default: all). E.g., easy medium hard",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the FAISS index",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_benchmark(
        experiment_name=args.experiment,
        dataset_names=args.datasets,
        force_rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
