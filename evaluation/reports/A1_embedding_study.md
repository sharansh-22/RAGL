# RAGL A1 — Embedding Model Study

**Objective**: Determine whether changing only the embedding model produces a measurable improvement in retrieval and generation quality.
**Baseline**: A0
**Candidates**: arctic, bge_base, bge_large, bge_small, e5_base, e5_large, nomic

## Constraints
- Chunk size and overlap remained identical to A0.
- The LLM (`llama3:8b`) remained identical.
- The FAISS index parameters remained identical.
- Comparison uses cached benchmark JSONs. No previous experiments were regenerated.

## Benchmark Comparison Table

| Model | hit_rate | mrr | ndcg | recall_at_k | precision_at_k | faithfulness | groundedness | relevancy | Latency (ms) |
|-------|---|---|---|---|---|---|---|---|--------------|
| A0 | 1.0000 | 1.0000 | 2.9485 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.9986 | 6363.1 |
| arctic | 0.8889 | 0.8333 | 2.1627 | 0.8333 | 0.7333 | 0.9333 | 0.1318 | 0.9983 | 8584.8 |
| bge_base | 0.8889 | 0.8333 | 2.3567 | 0.8333 | 0.8222 | 0.8803 | 0.0863 | 0.9991 | 7246.5 |
| bge_large | 0.8889 | 0.8889 | 2.3954 | 0.8333 | 0.8222 | 0.8889 | 0.1255 | 0.9992 | 38666.1 |
| bge_small | 0.8889 | 0.8148 | 2.2551 | 0.8333 | 0.7778 | 0.9370 | 0.0780 | 0.9989 | 30557.8 |
| e5_base | 0.8889 | 0.8056 | 2.2318 | 0.8333 | 0.7556 | 0.9454 | 0.2044 | 0.9986 | 6329.4 |
| e5_large | 0.8889 | 0.8333 | 2.3107 | 0.8333 | 0.8000 | 0.8497 | 0.2139 | 0.9966 | 14863.3 |
| nomic | 0.8889 | 0.8000 | 2.3196 | 0.8333 | 0.8000 | 0.9921 | 0.1444 | 0.9980 | 36865.9 |

## Engineering Recommendations

### 1. Retrieval Performance
Surprisingly, the **A0 Baseline** (`all-MiniLM-L6-v2`) outperformed all candidate models on this specific 9-query benchmark suite. 
A0 achieved a perfect **1.0000 Hit Rate** and **1.0000 MRR**, whereas every single candidate model in the A1 study scored a **0.8889 Hit Rate**, indicating they all missed exactly one query out of the 9.

### 2. Generation & Groundedness
A0 also maintained the highest `groundedness` score (0.5000) and perfect `faithfulness`. The candidate models struggled with groundedness, scoring between 0.07 and 0.21. 

### 3. Latency
The A0 baseline is extremely lightweight and fast (~6.3 seconds total average latency). The only model that matched its latency was `e5_base` (~6.3 seconds), but `e5_base` suffered from the same retrieval degradation as the other candidates. Models like `bge_large` and `nomic` experienced significantly higher latencies (30-38 seconds) with no improvement in accuracy.

### Final Verdict: Reject A1 Candidates
Based on the RAGL philosophy of controlled empirical evidence, there is no justification for switching the embedding model at this time. The 9-query dataset is small, but under identical conditions, none of the 7 candidate models demonstrated a measurable improvement over the baseline.

**Recommendation**: Retain the A0 Reference Implementation (`all-MiniLM-L6-v2`) as the permanent embedding model for future experiments. Do not promote any A1 candidate to the main branch.