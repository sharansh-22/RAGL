# Benchmark Bias Analysis (Validation Study V1)

## Overview
This report investigates potential sources of bias in the RAGL benchmark datasets that could artificially favor the A0 baseline or skew experiment results.

## Findings

### 1. Extremely Small Benchmark Size
- **Observation:** The entire benchmark consists of only **9 queries** across all difficulties (easy, medium, hard, math, coding, adversarial, long_context).
- **Impact:** A sample size of 9 is statistically insignificant for a robust evaluation framework. It causes high variance. A single query changing from a "miss" to a "hit" swings the Hit Rate by ~11%. This makes it difficult to reliably measure incremental improvements in new experiments.

### 2. Dataset Imbalance & Domain Skew
- **Observation:** The distribution of expected documents is highly skewed.
  - `mitres_18_001_f17_full_book.pdf` (Calculus text): 3 queries (33%)
  - `Software Engineering at Google.pdf`: 2 queries (22%)
  - `1706.03762v7.pdf` (Attention is All You Need): 2 queries (22%)
  - Remaining 2 PDFs only have 1 query each.
- **Impact:** Any architecture that happens to parse math textbooks slightly better will disproportionately dominate the overall benchmark score, masking poor performance on other domains like code or generic text.

### 3. Chunk Size Bias (Favoring A0)
- **Observation:** The queries in the dataset are predominantly factoid questions (e.g., "What is the chain rule in calculus?", "What is a neural network?").
- **Impact:** Factoid questions usually have localized answers that easily fit within A0's rigid 512-character chunk boundary. Because the benchmark lacks complex multi-hop reasoning questions or broad synthesis questions, it unintentionally favors the dense, uniform indexing approach of A0 over more sophisticated semantic or structural chunkers that excel at preserving long-form context.

### 4. Adversarial Query Handling
- **Observation:** `adversarial_01` correctly possesses an empty `expected_sources` array, meaning it tests the system's ability to abstain from hallucinating.
- **Impact:** While mathematically correct, with only 9 total queries, having 1 adversarial query means that abstention mechanics carry an 11% weight on the final Hit Rate and Faithfulness metrics.

## Conclusion
The benchmark is currently **biased toward A0** due to its small size and heavy reliance on simple, localized factoid queries. 

**Recommendation:** To conduct scientifically sound experiments in the future, the Gold Dataset must be expanded significantly (e.g., 50-100 queries) to dilute the variance and include a wider variety of query types (multi-hop, synthesis, comparative) to properly challenge the baseline architectures.
