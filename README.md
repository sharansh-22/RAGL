# RAGL — Retrieval-Augmented Generation Laboratory

## What is RAGL?

**RAGL** is a scientific framework for designing, benchmarking, and validating RAG architectures.

It is **not** a chatbot.  
It is **not** a product.  
It is a **research environment**.

Its sole objective is to discover, through controlled experimentation, what objectively improves a Retrieval-Augmented Generation system.

---

## Philosophy & Core Principles

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

## Scientific Workflow

Every architectural decision is made through a strict process:

1. **Reference**: Start from the current frozen baseline.
2. **Copy**: Duplicate the configuration into a new experiment (e.g., `A/A1/`).
3. **Modify**: Change exactly one variable.
4. **Benchmark**: Run the experiment against the evaluation datasets.
5. **Compare**: Analyze the metrics against the current baseline.
6. **Accept or Reject**: Conduct a human engineering review to make the final decision.
   - If accepted: The successor becomes the new Reference Implementation.
   - If rejected: The predecessor remains unchanged.

### Git Versioning

Every accepted experiment becomes a permanent Git milestone.

Suggested workflow:
`A0 -> git tag A0 -> A1 -> git tag A1 -> A2 -> git tag A2`

Every experiment should always remain reproducible. No accepted experiment should ever be overwritten.

### Baseline Preservation

The current accepted model becomes the Reference Implementation. It is frozen. Future work never modifies it directly.

Instead:
`Reference -> Copy -> Modify -> Benchmark -> Compare -> Accept or Reject`

- **If accepted**: The successor becomes the new Reference Implementation.
- **If rejected**: The predecessor remains unchanged.

---

## Repository Structure

```
RAGL/
├── A/                          # Controlled Experiments
│   └── A0/                     # Reference Implementations
│       ├── config.py           # Experiment constants
│       └── index/              # Cached FAISS index + chunks
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
│   ├── datasets/               # Evaluation query sets
│   ├── metrics/                
│   ├── reports/                # Engineering Reports
│   ├── plots/                  # Visualizations
│   └── cache/                  # Immutable evaluation cache
│
├── data/                       # Source documents
└── environment.yml             # Conda environment
```

*(Note: The separate PHILOSOPHY.md has been merged directly into this README to prevent redundancy.)*

---

## Experimental Pipeline & Documentation

Every completed experiment requires an Engineering Report inside `evaluation/reports/`.
Every report should include:

- **Objective**
- **Hypothesis**
- **Architecture**
- **Benchmark Results**
- **Visualizations**
- **Comparison with previous baseline**
- **Failure Analysis**
- **Engineering Review**
- **Final Decision**

The goal is that every architectural decision can be traced back to evidence.

### Human Engineering Review
Do not automatically accept or reject experiments based solely on metrics. **The evaluator's job is to measure. The engineer's job is to decide.** 

Every report must end with an Engineering Review that documents:
- Strengths
- Weaknesses
- Observed Behaviour
- Final Decision (Accepted / Rejected)
- Reason

For example, if a successor scores slightly higher numerically but produces significantly more verbose or lower-quality answers, it may still be rejected. Engineering judgment is a vital part of the scientific process.

### Evaluation Cache
All benchmark outputs are cached permanently inside `evaluation/cache/<experiment>/`. These cached outputs become immutable historical experiment records, storing: Query, Retrieved Documents, Retrieved Chunks, Prompt, Generated Answer, Retrieval Metrics, Generation Metrics, Latency, and Metadata. Experiments should only be rerun if the implementation changes.

---

## Current Reference Implementation

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
| **A0** | ✅ Reference | 1.000 | 1.000 | 2.948 | 1.000 | 1.000 | 1.000 | 0.500 | 0.999 | 6.4s | Accepted. Baseline established. |
| **A1** | Planned | - | - | - | - | - | - | - | - | - | Add BM25 sparse retrieval. |

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

---

## Roadmap

Upcoming controlled experiments:

- **A1**: Add BM25 sparse retrieval + Reciprocal Rank Fusion
- **A2**: Add FlashRank cross-encoder reranking
- **A3**: Compare embedding models (e5-large, instructor, etc.)
- **A4**: Vary chunk sizes and overlap strategies
- **A5**: Add multi-turn conversation context

