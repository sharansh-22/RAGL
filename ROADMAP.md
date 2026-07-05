# RAGL Future Roadmap

This document outlines the speculative, forward-looking experiments intended for RAGL. 

> [!WARNING]  
> **These are research directions, not promises.** Every single item on this list must survive objective evaluation and a strict engineering review before being adopted into the Reference Architecture.

## Upcoming Experiments

- **A4**: Add BM25 sparse retrieval + Reciprocal Rank Fusion (Hybrid Retrieval)
- **A5**: Query Transformation & Expansion (Hypothetical Document Embeddings, Step-Back Prompting)
- **A6**: Dynamic Context Assembly (Map-Reduce vs Refine generation chains)
- **A7**: Implementation of an LLM-as-a-Judge internal Verification Pipeline (Self-Correction)
- **A8**: Multi-Agent Routing (Query-Intent based retrieval branching)

## Dataset Expansion Goals

As concluded in the `V1` Evaluation Audit, the current benchmark dataset (5 queries) is far too small for robust statistical significance. 

Future dataset expansions will focus on:
- Adding at least 100 domain-specific difficult queries.
- Incorporating adversarial queries to test hallucination resistance.
- Including multi-hop reasoning queries to validate long-context retrieval algorithms.
