# RAGL — Retrieval-Augmented Generation Laboratory

## What is RAGL?

**RAGL** is a scientific framework for designing, benchmarking, and validating RAG architectures.

It is **not** a chatbot.  
It is **not** a product.  
It is a **research environment**.

Its sole objective is to discover, through controlled experimentation, what objectively improves a Retrieval-Augmented Generation system.

---

## Philosophy

The RAGL project operates under a strict scientific philosophy:

1. **One experiment changes exactly one variable.**
2. **Every experiment is reproducible.**
3. **Every experiment is benchmarked under identical conditions.**
4. **Every benchmark is cached and preserved.**
5. **A predecessor is replaced only when its successor demonstrates a meaningful and measurable improvement.**
6. **Metrics provide evidence.**
7. **Human engineering review makes the final decision.**
8. **Research-OS only receives components that have successfully graduated from RAGL.**

---

## Benchmark Once Compare Forever

RAGL enforces a strict policy regarding evaluation: **A completed experiment must never be rerun solely for comparison.**

Once an experiment has been benchmarked and accepted, its generated answers, benchmark metrics, reports, plots, and cached evidence become the canonical reference. Future experiments must compare against stored artifacts instead of regenerating previous experiments.

Previous experiments may only be rerun if:
- The implementation changes.
- The benchmark dataset changes.
- The evaluation framework changes in a way that invalidates previous metrics.
- A completely new baseline is intentionally being established.

Benchmarking and comparison are separate operations. Every accepted experiment becomes **Frozen**.

---

## Scientific Workflow

Every architectural decision is made through a strict process:

1. **Reference**: Start from the current frozen baseline.
2. **Copy**: Duplicate the configuration into a new experiment (e.g., `A/A1/`).
3. **Modify**: Change exactly one variable.
4. **Benchmark**: Run the experiment against the evaluation datasets.
5. **Compare**: Analyze the metrics against the current baseline using cached artifacts.
6. **Accept or Reject**: Conduct a human engineering review to make the final decision.
   - If accepted: The successor becomes the new Reference Implementation.
   - If rejected: The predecessor remains unchanged.

Every accepted experiment becomes a permanent Git milestone (`A0 -> git tag A0 -> A1 -> git tag A1`). No accepted experiment should ever be overwritten.

---

## Repository Structure

```
RAGL/
├── A/                          # Controlled Experiments
│   └── A0/                     # Reference Implementations
│       ├── config.py           # Experiment constants
│       ├── metadata.json       # Immutable experiment metadata
│       └── artifacts/          # Cached FAISS index + chunks
│
├── core/                       # Reusable RAG modules
│   ├── chunker.py              
│   ├── embedder.py             
│   ├── indexer.py              
│   ├── retriever.py            
│   ├── generator.py            
│   └── rag.py                  
│
├── evaluation/                 # Independent evaluation framework
│   ├── run.py                  # Benchmark runner
│   ├── compare.py              # Compares cached experiment artifacts
│   ├── datasets/               # Evaluation query sets
│   ├── metrics/                
│   ├── reports/                # Engineering Reports
│   ├── plots/                  # Visualizations
│   └── cache/                  # Immutable evaluation evidence (local only)
│
├── data/                       # Source documents
├── BASELINE.md                 # Current Reference Implementation details
└── environment.yml             # Conda environment
```

---

## Reference Implementation

**A0** is the baseline against which every future experiment is benchmarked. It implements the simplest possible RAG pipeline:

```text
Documents → Chunker → Embedder → FAISS Index → Retriever (Top-k) → Prompt Builder → llama3:8b → Answer
```

- **Embedding Model**: `BAAI/bge-small-en-v1.5` (384-dim)
- **LLM**: `llama3:8b` (Ollama)
- **Chunk Size / Overlap**: ~512 / ~64 tokens (Sentence-aware recursive)
- **Retrieval**: Top-5 FAISS IndexFlatIP

A0 **does not** include BM25, hybrid retrieval, rerankers, planners, intent classifiers, multi-LLM routing, or agents. These belong to future experiments.

---

## Model History

The Model History is the permanent historical record of every completed experiment in RAGL. Rejected experiments are just as valuable as accepted ones.

| Version | Status | Hit Rate | MRR | NDCG | P@K | R@K | Faith | Ground | Relevancy | Latency | Decision |
|---------|--------|----------|-----|------|-----|-----|-------|--------|-----------|---------|----------|
| **A0** | ✅ Reference | 1.000 | 1.000 | 2.948 | 1.000 | 1.000 | 1.000 | 0.500 | 0.999 | 6.4s | Accepted. Baseline established (2 queries). |
| **A1** | ❌ Rejected | 0.888 | 0.888 | 2.395 | 0.822 | 0.833 | 0.888 | 0.125 | 0.999 | 38.6s | Rejected. Baseline `bge-small` retained. No candidate justified the latency cost. |
| **A2** | ❌ Rejected | 0.888 | 0.833 | 2.267 | 0.778 | 0.833 | 0.989 | 0.222 | 0.997 | 3.8s | Rejected. Baseline retained, but `structure` chunking improved latency and brevity. |

---

## Current Benchmark

Our current reference baseline (**A0**) scored:
- **Hit Rate**: 1.000
- **Recall@K**: 1.000
- **Precision@K**: 1.000
- **MRR**: 1.000
- **NDCG**: 2.948
- **Faithfulness**: 1.000
- **Groundedness**: 0.500
- **Relevancy**: 0.999
- **Latency**: ~6.4s

*All benchmark cache outputs are stored locally in `evaluation/cache/A0/` as immutable scientific evidence.*

---

## Roadmap

Upcoming controlled experiments:

- **A1**: Embedding Model Selection Study (Completed - Rejected)
- **A2**: Chunking Strategy Selection (Completed - Rejected)
- **A3**: Add BM25 sparse retrieval + Reciprocal Rank Fusion
- **A4**: Add FlashRank cross-encoder reranking
- **A5**: Add multi-turn conversation context

---

## Relationship with Research-OS

RAGL and Research-OS operate as a tandem ecosystem with a strict one-way flow of validated knowledge and architecture:

```text
RAGL
  ↓
Validated Components
  ↓
Research-OS
  ↓
Production AI Research Assistant
```

**Research-OS** is the production implementation built *exclusively* from architectures validated inside RAGL. It should never be used as an experimentation playground. The success of RAGL is measured by the quality of its methodology, reproducibility, and evidence. The success of Research-OS is measured by the quality of the final user experience. 
