# RAGL A2 — Chunking Strategy Study

**Objective**: Determine whether changing the chunking strategy produces a measurable improvement in retrieval and generation quality.
**Baseline**: A0
**Candidates**: recursive, semantic, sentence, structure

## Constraints
- Comparison uses cached benchmark JSONs. No previous experiments were regenerated.

## Benchmark Comparison Table

| Model | hit_rate | mrr | ndcg | recall_at_k | precision_at_k | faithfulness | groundedness | relevancy | avg_retrieved_context_len | Latency (ms) |
|-------|---|---|---|---|---|---|---|---|---|--------------|
| A0 | 1.0000 | 1.0000 | 2.9485 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.9986 | N/A | 6363.1 |
| recursive | 0.8889 | 0.8333 | 2.2934 | 0.8333 | 0.8000 | 0.8276 | 0.1233 | 0.9976 | 2964.7556 | 6661.6 |
| semantic | 0.8889 | 0.8148 | 2.2142 | 0.8333 | 0.7556 | 0.7500 | 0.3231 | 0.9996 | 343.1778 | 4046.4 |
| sentence | 0.8889 | 0.8148 | 2.2551 | 0.8333 | 0.7778 | 0.9506 | 0.1287 | 0.9988 | 3084.4667 | 7985.6 |
| structure | 0.8889 | 0.8333 | 2.2670 | 0.8333 | 0.7778 | 0.9889 | 0.2222 | 0.9970 | 1415.2667 | 3885.5 |

## Chunking Statistics

| Model | Total Chunks | Avg Size | Med Size | Max Size | Min Size | Overlap | Index Size (KB) | Index Time (s) |
|-------|--------------|----------|----------|----------|----------|---------|-----------------|----------------|
| A0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| recursive | 2752 | 2793.4 | 2855.0 | 8646.0 | 4.0 | 64 | 4128.0 | 0.00 |
| semantic | 53782 | 130.9 | 89.0 | 5017.0 | 1.0 | 64 | 80673.0 | 0.03 |
| sentence | 2669 | 2952.0 | 2963.0 | 8857.0 | 19.0 | 64 | 4003.5 | 0.00 |
| structure | 5001 | 1473.0 | 1305.0 | 8485.0 | 4.0 | 64 | 7501.5 | 0.00 |

## Engineering Recommendations

> **PENDING**: Human Engineering Review will be appended here after qualitative inspection.

### Qualitative Engineering Review

1. **A0 Baseline Dominance**: A0 significantly outperformed all A2 strategies on retrieval metrics (1.0 hit rate vs 0.8889). A0's naive fixed-size chunking (using Langchain's defaults) seems to strike an optimal balance for this specific dataset and query set, likely due to consistent context lengths preventing the retriever from being biased towards extremely short or long chunks.
2. **Semantic Chunking Is Granular**: The `semantic` strategy produced an enormous number of chunks (53,782) with a very small average size (~130 chars). This resulted in an 80MB index and an extremely short `avg_retrieved_context_len` (343 chars). While generation latency improved (4.0s) due to the small prompt, the context was arguably too fragmented.
3. **Structure Strategy Efficiency**: `structure` chunking (Markdown-aware) yielded 5,001 chunks (avg size ~1473 chars) and provided the lowest overall latency (3.8s) among all A2 strategies. It maintained high faithfulness (0.9889) and precision, suggesting that keeping section headings intact helps the LLM generate accurate answers with less reading overhead.
4. **Conclusion**: While `structure` chunking provides an excellent trade-off for latency and faithfulness, it did not strictly beat the A0 baseline's retrieval performance on this specific benchmark. **A2 is rejected** in terms of absolute retrieval metrics, but `structure` chunking is recommended for future iterations where latency and context brevity are priorities.