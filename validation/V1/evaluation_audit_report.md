# Evaluation Framework Audit Report (Validation Study V1)

## Executive Summary
This report summarizes the findings of Validation Study V1. The objective was to audit the RAGL evaluation framework (retrieval metrics, generation metrics, and datasets) to ensure mathematical correctness, cross-metric consistency, and lack of bias.

## 1. Retrieval Metrics Audit
**Result: PASSED**

A synthetic unit test (`audit_retrieval.py`) was constructed to manually compute metrics for edge cases (perfect match, partial match, out-of-order, duplicates, empty datasets) using strict mathematical formulas.
- **Recall@K**: Correctly implemented as $|R_k \cap E| / |E|$
- **Precision@K**: Correctly implemented as $|R_k \cap E| / k$
- **Hit Rate**: Correctly implemented as binary $1$ if $|R \cap E| > 0$ else $0$.
- **MRR**: Correctly implemented as $1 / \text{rank}$.
- **NDCG**: Correctly implemented using standard DCG discounting ($1 / \log_2(i+2)$) and Ideal DCG normalization.

All programmatic outputs perfectly matched manual calculations. The mathematical foundation of the retrieval evaluator is entirely sound.

## 2. Generation Metrics Audit
**Result: PASSED**

A synthetic unit test (`audit_generation.py`) assessed the Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) and NLI (`nli-deberta-v3-small`) models against controlled adversarial inputs.
- **Faithfulness (NLI)**: Successfully detected and penalized hallucinations. A perfect answer scored 1.0, while introducing a hallucinated sentence caused the score to drop proportionately as the NLI model correctly classified the hallucinated claim as a contradiction.
- **Groundedness (NLI)**: Correctly verified that individual sentences are entailed by specific retrieved chunks. The multi-chunk logic successfully avoids context truncation issues.
- **Relevancy (Cross-Encoder)**: Correctly distinguished between highly relevant answers (0.999 score) and completely irrelevant answers (0.00001 score). Sigmoid normalization is functioning smoothly.

## 3. Retrieval Evidence & Cross-Metric Consistency
**Result: PASSED**

A review of past benchmark outputs confirms that metrics correlate logically. In the A2 experiment, `semantic` chunking yielded massive index sizes and tiny chunks, which mathematically resulted in very low Groundedness scores (~0.32). This proves the evaluator can correctly penalize architectures that destroy contextual integrity. There are no suspicious states (e.g., perfect relevancy with 0.0 hit rate) detected in the benchmark caches.

## 4. Benchmark Bias Audit
**Result: FAILED (Bias Detected)**

While the evaluation codebase is mathematically correct, the underlying **Gold Dataset** contains significant flaws that threaten the integrity of the benchmark:
1. **Insignificant Sample Size**: The dataset contains only 9 queries. A single random hallucination swings overall metrics by ~11%, making it impossible to measure fine-grained architectural improvements.
2. **Domain Skew**: 33% of the dataset focuses entirely on a single Calculus textbook.
3. **Query Bias**: The queries are exclusively simple factoid questions. Because these answers are naturally short, they heavily favor the A0 baseline (512-character fixed chunks). The benchmark lacks multi-hop or synthesis questions required to justify advanced chunking or retrieval strategies.

*(For full details, see `benchmark_bias_analysis.md`).*

## Final Conclusion

**Option 2**

The evaluation framework contains issues that must be corrected before conducting additional RAG experiments. While the mathematical implementation of the metrics is scientifically sound, the Gold Dataset is fundamentally biased and statistically insignificant. The identified issues are documented, and no further architecture experiments should proceed until the dataset is expanded and diversified.
