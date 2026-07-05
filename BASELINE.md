# RAGL Reference Architecture v1.0

Whenever a future experiment replaces the current baseline, this file is updated to reflect the new reference implementation.

## Current Reference Implementation: A0
- **Status:** Frozen
- **Version:** v1.0
- **Experiment Date:** July 2026

## Core Architecture
- **Embedding Model:** `BAAI/bge-small-en-v1.5`
- **LLM:** `llama3:8b`
- **Chunk Size:** 512
- **Chunk Overlap:** 64
- **Top-k:** 5
- **Chunking Strategy:** Sentence-Aware Recursive
- **Retrieval:** FAISS IndexFlatIP (Dense)

## Current Benchmark Metrics (A0 Baseline - 2 Queries)
- **Hit Rate**: 1.000
- **Recall@K**: 1.000
- **Precision@K**: 1.000
- **MRR**: 1.000
- **NDCG**: 2.948
- **Faithfulness**: 1.000
- **Groundedness**: 0.500
- **Relevancy**: 0.999
- **Latency**: ~6.4s

> [!NOTE]  
> The original A0 baseline was established on a 2-query dataset. As discovered in `V1` and `A3`, running the exact same pipeline on a 5-query dataset yields an MRR of 0.8667. Expanding the dataset is a priority for future evaluations.
