# A0 — Benchmark Report

**Generated**: 2026-07-04 19:51:47
**Experiment**: A0 (Reference Implementation)
**Queries evaluated**: 2

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| LLM Model | `llama3:8b` |
| Chunk Size | 512 |
| Chunk Overlap | 64 |
| Top-K | 5 |

## Retrieval Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| hit_rate | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| mrr | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| ndcg | 2.9485 | 0.0000 | 2.9485 | 2.9485 |
| precision_at_k | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| recall_at_k | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

## Generation Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| faithfulness | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| groundedness | 0.5000 | 0.5000 | 0.0000 | 1.0000 |
| relevancy | 0.9986 | 0.0005 | 0.9982 | 0.9991 |

## Performance

| Stage | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|-------|-----------|----------|----------|----------|
| retrieval_ms | 7.4 | 3.4 | 4.0 | 10.8 |
| generation_ms | 6355.7 | 1312.3 | 5043.4 | 7667.9 |
| total_ms | 6363.1 | 1308.9 | 5054.2 | 7672.0 |

## Per-Query Results

### easy_01
**Query**: What is the chain rule in calculus?
**Difficulty**: easy
**Category**: math

Retrieval: hit_rate=1.0000, mrr=1.0000, ndcg=2.9485, precision_at_k=1.0000, recall_at_k=1.0000
Generation: faithfulness=1.0000, groundedness=1.0000, relevancy=0.9991
Latency: retrieval_ms=4.0ms, generation_ms=7667.9ms, total_ms=7672.0ms

### easy_02
**Query**: What is a neural network?
**Difficulty**: easy
**Category**: machine_learning

Retrieval: hit_rate=1.0000, mrr=1.0000, ndcg=2.9485, precision_at_k=1.0000, recall_at_k=1.0000
Generation: faithfulness=1.0000, groundedness=0.0000, relevancy=0.9982
Latency: retrieval_ms=10.8ms, generation_ms=5043.4ms, total_ms=5054.2ms
