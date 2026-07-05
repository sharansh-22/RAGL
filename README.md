# RAGL — Retrieval-Augmented Generation Laboratory

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Release](https://img.shields.io/badge/Release-v1.0.0-blueviolet)
![Status: Stable](https://img.shields.io/badge/Status-Stable-success)
![CUDA Tested](https://img.shields.io/badge/CUDA-12_Tested-orange)

## What is RAGL?

**RAGL** is a scientific framework for designing, benchmarking, and validating RAG architectures.

Most personal or enterprise RAG projects continuously add new features (e.g., rerankers, agents, graph databases) without validating whether they actually improve the architecture. RAGL exists to solve that problem.

It is **not** a chatbot.  
It is **not** a product.  
It is a **research environment**.

Every architectural decision is made through controlled experiments. The objective is not to build the most complex RAG. The objective is to identify the simplest architecture that survives objective evaluation.

---

## Philosophy

The RAGL project operates under a strict scientific engineering philosophy:

1. **One Variable Per Experiment**: Never change the chunker and the embedder at the same time. Isolate variables to measure their true impact.
2. **Frozen Baselines**: The current Reference Architecture (e.g., `A0`) is immutable. Never overwrite it.
3. **Reproducibility**: Ensure your experiment can be run by anyone using the existing datasets and scripts.
4. **Benchmark Before Adoption**: No architecture is promoted without objective evidence showing meaningful improvement over the baseline.
5. **Evidence Over Assumptions**: Do not assume that a more complex architecture is better. Prove it.
6. **Simplicity Over Complexity**: Architectural changes must justify their computational and latency cost.
7. **Human Engineering Review**: Metrics guide us, but human review makes the final decision.

---

## Experimental Workflow

Every architectural decision is made through a strict process:

```text
Hypothesis
   ↓
Implement
   ↓
Benchmark
   ↓
Evaluation
   ↓
Engineering Review
   ↓
Accept / Reject
   ↓
Reference Architecture
```

If an experiment is **Accepted**, it becomes the new Reference Architecture. If it is **Rejected**, the baseline remains unchanged.

---

## Repository Structure

```
RAGL/
├── A/                          # Controlled Experiments
│   └── A0/                     # Reference Architecture v1.0
│
├── core/                       # Reusable RAG modules
│   ├── chunker.py              
│   ├── embedder.py             
│   ├── indexer.py              
│   ├── retriever.py            
│   ├── generator.py            
│   └── rag.py                  
│
├── evaluation/                 # Independent Evaluation Framework
│   ├── run.py                  # Benchmark runner
│   ├── compare.py              # Compares cached experiment artifacts
│   ├── datasets/               # Evaluation query sets
│   ├── metrics/                # Metric computations
│   ├── reports/                # Final Engineering Reports
│   └── cache/                  # Immutable evaluation evidence
│
├── validation/                 # Internal Framework Audits
│   └── V1/                     # Evaluator Validation Audit
│
├── data/                       # Source documents
└── docs/                       # Additional documentation
```

---

## Glossary

- **Reference Architecture**: The current best-performing, accepted architecture that powers Research-OS (e.g., `A0`).
- **Benchmark**: The process of running the architecture against a fixed dataset to generate metrics.
- **Evidence**: The cached artifacts, metric outputs, and JSON logs proving the performance of an experiment.
- **Validation**: The process of ensuring the evaluation framework itself is trustworthy (e.g., `V1`).
- **Promotion**: The act of accepting a successful experiment and making it the new Reference Architecture.
- **Experiment**: A strictly isolated modification to the architecture to test a single hypothesis (e.g., `A1`, `A2`).

---

## Reference Architecture v1.0 (A0)

**A0** is the baseline against which every future experiment is benchmarked. It implements the simplest possible RAG pipeline:

```text
Documents → Chunker → Embedder → FAISS Index → Retriever (Top-k) → Prompt Builder → llama3:8b → Answer
```

- **Embedding Model**: `BAAI/bge-small-en-v1.5` (384-dim)
- **LLM**: `llama3:8b` (Ollama)
- **Chunk Size / Overlap**: ~512 / ~64 tokens (Sentence-aware recursive)
- **Retrieval**: Top-5 FAISS IndexFlatIP

See `ARCHITECTURE.md` and `BASELINE.md` for more details.

---

## Experimental History

The Model History is the permanent historical record of every completed experiment in RAGL. Rejected experiments are just as valuable as accepted ones.

| Experiment | Objective | Result | Decision |
|------------|-----------|--------|----------|
| **A0** | Establish Baseline | Passed | ✅ Reference |
| **A1** | Evaluate Embedding Models | No Improvement | ❌ Rejected |
| **A2** | Evaluate Chunking Strategies | No Improvement | ❌ Rejected |
| **A3** | Evaluate Rerankers | Poor ROI | ❌ Rejected |
| **V1** | Audit Evaluation Framework | Passed | ✅ Validated |

### Engineering Decisions

| Component | Decision | Rationale |
|-----------|----------|-----------|
| **Rerankers (A3)** | Rejected | While CrossEncoders provided a +0.033 MRR boost, they degraded overall NDCG, dropped faithfulness, and added 100-2500ms of latency overhead. The ROI was too poor to justify architectural complexity. |
| **Semantic Chunking (A2)** | Rejected | Added 23 minutes to ingestion time with zero improvement to benchmark metrics compared to Sentence-Aware Recursive chunking. |
| **Large Embeddings (A1)** | Rejected | Larger models (`e5-large`, `bge-large`) produced identical Top-5 hit rates as `bge-small` but increased FAISS memory footprint and inference latency. |

### Lessons Learned
- **Complexity does not imply better performance.**
- Simple architectures often outperform sophisticated ones on smaller, domain-specific corpora.
- Architectural changes must always justify their computational cost.
- Every accepted feature must survive objective evaluation.

---

## Evaluation Framework: V1 (Validated)

RAGL enforces a strict policy regarding evaluation: **A completed experiment must never be rerun solely for comparison.**

Our evaluation metrics (Hit Rate, MRR, NDCG, Faithfulness, Groundedness) and the mathematical formulas backing them were fully audited during the **V1 Validation Study**.

### Benchmark Limitations
- **Dataset Size**: The current benchmark consists of a very small (5-query) dataset spanning multiple domains. Future iterations will focus on expanding this to 100+ queries to improve statistical significance.
- **Hardware Limitations**: Benchmarks were run on consumer hardware, leading to varying latencies for LLM generation.
- **Corpus Constraints**: The validation corpus currently spans only three computer science textbooks.

---

## Relationship with Research-OS

RAGL and Research-OS operate as a tandem ecosystem with a strict one-way flow of validated knowledge and architecture:

```text
Experiments (RAGL)
   ↓
Reference Architecture (RAGL)
   ↓
Research-OS Application
```

**RAGL is the laboratory.**  
**Research-OS is the application.**

Research-OS is built *exclusively* from the current Reference Architecture validated inside RAGL. *Powered by RAGL Reference Architecture v1.0 (A0).*

---

## References

- **Embeddings**: [FastEmbed](https://qdrant.github.io/fastembed/)
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss)
- **Local Generation**: [Ollama](https://ollama.com/) & Llama 3
- **Rerankers**: [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
