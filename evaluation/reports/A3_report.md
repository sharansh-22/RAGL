# Experiment A3: Reranker Evaluation Study

## Objective
Determine whether adding a reranking stage to the A0 retrieval pipeline produces a measurable improvement in retrieval and generation quality, and whether it should become a permanent architectural component of Research-OS.

## The A0 Baseline Discrepancy
The raw `A0` baseline achieved an MRR of 1.000, while the `A0_No_Reranker` baseline (which utilizes the exact same retrieval pipeline) achieved an MRR of 0.8667. 

**Why?**
The `A0` cached benchmark only contains results for **2 queries** (`easy_01`, `easy_02`), both of which were perfect hits. The `A3` pipeline evaluated against the full 5-query dataset (including `medium` and `hard` queries). Therefore, `A0_No_Reranker` serves as the true baseline for this study, as it evaluates the A0 architecture against the exact same 5 queries.

## Results Summary

| Model | MRR | NDCG | Recall@5 | Precision@5 | Latency (ms) |
|-------|---|---|---|---|--------------|
| **A0_No_Reranker** (Baseline) | 0.8667 | 2.3902 | 0.9000 | 0.8400 | ~5500 |
| **CrossEncoder_MiniLM** | 0.9000 | 2.3201 | 0.9000 | 0.8000 | ~6700 |
| **BGE_Reranker_Base** | 0.9000 | 2.2726 | 0.9000 | 0.7600 | ~8000 |
| **FlashRank_MiniLM** | 0.8667 | 2.1392 | 0.9000 | 0.7200 | ~6300 |

### Key Observations
1. **Marginal MRR Improvement**: `CrossEncoder` and `BGE` slightly improved MRR (0.8667 → 0.9000).
2. **NDCG Degradation**: All rerankers degraded NDCG compared to the baseline.
3. **Latency Penalty**: Rerankers added noticeable latency. `BGE_Reranker_Base` added nearly 2.5 seconds to total pipeline execution, while `CrossEncoder_MiniLM` and `FlashRank` added ~800-1200ms.
4. **Generation Metrics**: Reranking slightly degraded generation `faithfulness` and `groundedness`.

## Qualitative Analysis (`retrieved_before` vs `retrieved_after`)

By analyzing the `hard_01` query ("Derive the backpropagation update rule..."):
- **Before (A0)**: The first highly relevant chunk ("Bishop - Pattern Recognition") surfaced at **Rank 3**, resulting in an MRR of 0.33.
- **After (CrossEncoder)**: The reranker successfully identified this chunk and promoted it to **Rank 2**, increasing MRR to 0.50.
- **The Tradeoff**: While the single most relevant chunk was promoted, the reranker scrambled the order of other contextually relevant chunks. Because NDCG penalizes misorderings across the *entire* top-k set, the overall NDCG dropped despite the MRR gain. 
- **Generation Impact**: By promoting a partially relevant mathematical chunk, the LLM attempted to answer the difficult question but hallucinated (dropping `faithfulness` from 1.0 to 0.85). Without the reranker, the LLM safely admitted it lacked context (`faithfulness` = 1.0).

## Engineering Decision

> [!WARNING]  
> **REJECTED: Do NOT make reranking a permanent architectural component at this time.**

**Rationale:**
1. **Poor ROI**: A +0.033 gain in MRR does not justify a 15-40% increase in total latency and the added architectural complexity of deploying cross-encoder models.
2. **Harmful to Holistic Quality**: The degradation in NDCG indicates that rerankers are optimizing for single-chunk relevance at the expense of overall context cohesion.
3. **Statistical Insignificance**: As concluded in Validation Study V1, the dataset currently consists of only 5 queries. Making permanent, complex architectural changes based on a 5-query sample is dangerous engineering practice. 

**Next Steps**: Focus on expanding the benchmark dataset (as recommended in V1) before attempting to squeeze marginal gains out of the retrieval pipeline.
