# A1 — Benchmark Report

**Generated**: 2026-07-05 13:11:45
**Experiment**: A1 (Reference Implementation)
**Queries evaluated**: 9

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Model | `Snowflake/snowflake-arctic-embed-m-v1.5` |
| LLM Model | `llama3:8b` |
| Chunk Size | 512 |
| Chunk Overlap | 64 |
| Top-K | 5 |

## Retrieval Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| hit_rate | 0.8889 | 0.3143 | 0.0000 | 1.0000 |
| mrr | 0.8333 | 0.3333 | 0.0000 | 1.0000 |
| ndcg | 2.1627 | 1.1613 | 0.0000 | 2.9485 |
| precision_at_k | 0.7333 | 0.3887 | 0.0000 | 1.0000 |
| recall_at_k | 0.8333 | 0.3333 | 0.0000 | 1.0000 |

## Generation Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| faithfulness | 0.9333 | 0.1886 | 0.4000 | 1.0000 |
| groundedness | 0.1318 | 0.2577 | 0.0000 | 0.8000 |
| relevancy | 0.9983 | 0.0022 | 0.9932 | 1.0000 |

## Performance

| Stage | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|-------|-----------|----------|----------|----------|
| retrieval_ms | 10.7 | 3.9 | 6.1 | 17.8 |
| generation_ms | 8574.0 | 3455.4 | 5462.3 | 15878.7 |
| total_ms | 8584.8 | 3456.0 | 5477.1 | 15888.0 |

## Per-Query Results

### adversarial_01
**Query**: What does the textbook say about quantum computing applications in machine learning?
**Difficulty**: hard
**Category**: adversarial

Retrieval: hit_rate=0.0000, mrr=0.0000, ndcg=0.0000, precision_at_k=0.0000, recall_at_k=0.0000
Generation: faithfulness=0.4000, groundedness=0.0000, relevancy=0.9970
Latency: retrieval_ms=7.4ms, generation_ms=6460.6ms, total_ms=6468.1ms

### coding_01
**Query**: What are the best practices for writing unit tests in a large-scale software project?
**Difficulty**: medium
**Category**: software_engineering

Retrieval: hit_rate=1.0000, mrr=1.0000, ndcg=2.9485, precision_at_k=1.0000, recall_at_k=1.0000
Generation: faithfulness=1.0000, groundedness=0.0000, relevancy=0.9999
Latency: retrieval_ms=14.8ms, generation_ms=5462.3ms, total_ms=5477.1ms

### easy_01
**Query**: What is the chain rule in calculus?
**Difficulty**: easy
**Category**: math

Retrieval: hit_rate=1.0000, mrr=1.0000, ndcg=2.9485, precision_at_k=1.0000, recall_at_k=1.0000
Generation: faithfulness=1.0000, groundedness=0.8000, relevancy=0.9963
Latency: retrieval_ms=8.1ms, generation_ms=5903.6ms, total_ms=5911.8ms

### easy_02
**Query**: What is a neural network?
**Difficulty**: easy
**Category**: machine_learning

Retrieval: hit_rate=1.0000, mrr=1.0000, ndcg=1.3869, precision_at_k=0.4000, recall_at_k=1.0000
Generation: faithfulness=1.0000, groundedness=0.0000, relevancy=0.9989
Latency: retrieval_ms=6.1ms, generation_ms=8945.8ms, total_ms=8951.9ms

### hard_01
**Query**: Derive the backpropagation update rule for a single hidden layer neural network using the chain rule and explain how gradient flow relates to vanishing gradients.
**Difficulty**: hard
**Category**: machine_learning

Retrieval: hit_rate=1.0000, mrr=0.5000, ndcg=0.3869, precision_at_k=0.2000, recall_at_k=0.5000
Generation: faithfulness=1.0000, groundedness=0.0526, relevancy=0.9993
Latency: retrieval_ms=9.3ms, generation_ms=15878.7ms, total_ms=15888.0ms

### long_context_01
**Query**: Provide a comprehensive summary of the multi-head attention mechanism including the mathematical formulation of queries, keys, and values, the scaling factor, and how multiple heads are concatenated and projected.
**Difficulty**: hard
**Category**: machine_learning

Retrieval: hit_rate=1.0000, mrr=1.0000, ndcg=2.9485, precision_at_k=1.0000, recall_at_k=1.0000
Generation: faithfulness=1.0000, groundedness=0.0000, relevancy=0.9932
Latency: retrieval_ms=17.8ms, generation_ms=13547.6ms, total_ms=13565.4ms

### math_01
**Query**: What is the fundamental theorem of calculus and how does it connect differentiation and integration?
**Difficulty**: medium
**Category**: math

Retrieval: hit_rate=1.0000, mrr=1.0000, ndcg=2.9485, precision_at_k=1.0000, recall_at_k=1.0000
Generation: faithfulness=1.0000, groundedness=0.3333, relevancy=0.9999
Latency: retrieval_ms=15.2ms, generation_ms=6513.7ms, total_ms=6528.9ms

### medium_01
**Query**: Explain the attention mechanism in transformers and why it replaced recurrence.
**Difficulty**: medium
**Category**: machine_learning

Retrieval: hit_rate=1.0000, mrr=1.0000, ndcg=2.9485, precision_at_k=1.0000, recall_at_k=1.0000
Generation: faithfulness=1.0000, groundedness=0.0000, relevancy=0.9999
Latency: retrieval_ms=8.5ms, generation_ms=7414.9ms, total_ms=7423.4ms

### medium_02
**Query**: What are the key principles of code review according to software engineering best practices?
**Difficulty**: medium
**Category**: software_engineering

Retrieval: hit_rate=1.0000, mrr=1.0000, ndcg=2.9485, precision_at_k=1.0000, recall_at_k=1.0000
Generation: faithfulness=1.0000, groundedness=0.0000, relevancy=1.0000
Latency: retrieval_ms=9.2ms, generation_ms=7039.0ms, total_ms=7048.2ms
