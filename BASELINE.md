# RAGL Reference Baseline

Whenever a future experiment replaces the current baseline, only this file is updated to reflect the new reference implementation.

## Current Reference Implementation: A0
- **Status:** Frozen
- **Version:** 0.1
- **Experiment Date:** July 2026

## Core Architecture
- **Embedding Model:** `BAAI/bge-small-en-v1.5`
- **LLM:** `llama3:8b`
- **Chunk Size:** 512
- **Chunk Overlap:** 64
- **Top-k:** 5

## Current Benchmark Metrics
- **Hit Rate**: 1.000
- **Recall@K**: 1.000
- **Precision@K**: 1.000
- **MRR**: 1.000
- **NDCG**: 2.948
- **Faithfulness**: 1.000
- **Groundedness**: 0.500
- **Relevancy**: 0.999
- **Latency**: ~6.4s
