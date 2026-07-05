"""
A3 Reranker Study Runner
"""
import sys
from pathlib import Path
import json
import numpy as np

# Ensure RAGL root is on the path
RAGL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAGL_ROOT))

from A.A3.config import (
    CANDIDATES, EMBEDDING_MODEL, LLM_MODEL, TOP_K, RETRIEVE_TOP_K,
    INDEX_DIR, EVAL_DATASETS_DIR, EVAL_CACHE_DIR
)
from core.embedder import Embedder
from core.indexer import load_index
from core.rag_rerank import RerankRAGPipeline
from evaluation.run import load_datasets
from evaluation.metrics.retrieval import compute_all_retrieval_metrics
from evaluation.metrics.generation import compute_all_generation_metrics
from core.rerankers.flashrank_reranker import FlashRankReranker
from core.rerankers.cross_encoder_reranker import CrossEncoderReranker

from rich.console import Console
console = Console()

def get_reranker(strategy: str, model_name: str = None):
    if strategy == "none":
        return None
    elif strategy == "flashrank":
        return FlashRankReranker(model_name=model_name)
    elif strategy == "cross_encoder":
        return CrossEncoderReranker(model_name=model_name)
    else:
        raise ValueError(f"Unknown reranker strategy {strategy}")

def run_candidate(candidate_name: str, candidate_config: dict):
    console.print(f"\\n[bold green]Running Candidate: {candidate_name}[/bold green]")
    
    # Load A0 index strictly
    index_path = Path(RAGL_ROOT) / INDEX_DIR / "faiss.index"
    chunks_path = Path(RAGL_ROOT) / INDEX_DIR / "chunks.json"
    
    embedder = Embedder(model_name=EMBEDDING_MODEL)
    index = load_index(str(index_path))
    with open(chunks_path) as f:
        chunks = json.load(f)
        
    reranker = get_reranker(candidate_config["strategy"], candidate_config.get("model_name"))
    
    pipeline = RerankRAGPipeline(
        embedder=embedder,
        index=index,
        chunks=chunks,
        model_name=LLM_MODEL,
        retrieve_top_k=RETRIEVE_TOP_K,
        final_top_k=TOP_K,
        reranker=reranker
    )
    
    queries = load_datasets(str(Path(RAGL_ROOT) / EVAL_DATASETS_DIR), ["easy", "medium", "hard"])
    results = []
    
    for i, q in enumerate(queries, start=1):
        console.print(f"  [{i}/{len(queries)}] {q['id']}")
        
        result = pipeline.query(q["query"])
        
        ret_metrics = compute_all_retrieval_metrics(
            result["retrieved_chunks"],
            q.get("expected_sources", []),
            k=TOP_K
        )
        
        gen_metrics = compute_all_generation_metrics(
            answer=result["answer"],
            query=q["query"],
            context=result["retrieved_chunks"]
        )
        
        full_result = {
            "id": q["id"],
            "query": q["query"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "expected_sources": q.get("expected_sources", []),
            "retrieved_chunks": result["retrieved_chunks"],
            "retrieved_sources": list(set(c["source"] for c in result["retrieved_chunks"])),
            "retrieved_before": result["retrieved_before"],
            "retrieved_after": result["retrieved_after"],
            "prompt": result["prompt"],
            "answer": result["answer"],
            "latency": result["latency"],
            "retrieval_metrics": ret_metrics,
            "generation_metrics": gen_metrics,
        }
        results.append(full_result)
        
        console.print(f"         hit={ret_metrics['hit_rate']:.0f} lat={result['latency']['total_ms']:.0f}ms")
        
    # Save cache
    cache_dir = Path(RAGL_ROOT) / EVAL_CACHE_DIR / candidate_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for r in results:
        query_file = cache_dir / f"{r['id']}.json"
        
        level2_evidence = {
            "experiment": "A3",
            "candidate": candidate_name,
            "configuration": {
                "embedding_model": EMBEDDING_MODEL,
                "llm": LLM_MODEL,
                "top_k": TOP_K,
                "retrieve_top_k": RETRIEVE_TOP_K,
                "reranker": candidate_config
            },
            "query_id": r["id"],
            "query": r["query"],
            "prompt": r["prompt"],
            "answer": r["answer"],
            "retrieved_documents": r["retrieved_sources"],
            "retrieved_before": r["retrieved_before"],
            "retrieved_after": r["retrieved_after"],
            "retrieved_chunk_ids": [c.get("id", i) for i, c in enumerate(r["retrieved_chunks"])],
            "retrieval_metrics": r["retrieval_metrics"],
            "generation_metrics": r["generation_metrics"],
            "latency_ms": r["latency"].get("total_ms"),
            "latency_breakdown": r["latency"]
        }
        
        with open(query_file, "w") as f:
            json.dump(level2_evidence, f, indent=2, default=str)
            
    # Save summary
    summary = {
        "experiment": "A3",
        "candidate": candidate_name,
        "num_queries": len(results),
        "results_files": [f"{r['id']}.json" for r in results]
    }
    with open(cache_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def main():
    for candidate_name, config in CANDIDATES.items():
        run_candidate(candidate_name, config)
        
if __name__ == "__main__":
    main()
