# Contributing to RAGL

RAGL is a scientific laboratory for evaluating Retrieval-Augmented Generation architectures. If you wish to contribute an experiment, please adhere strictly to our engineering philosophy.

## The RAGL Philosophy

1. **One Variable Per Experiment**: Never change the chunker and the embedder at the same time. Isolate variables to measure their true impact.
2. **Frozen Baselines**: The current Reference Architecture (e.g., `A0`) is immutable. Never overwrite it.
3. **Reproducibility**: Ensure your experiment can be run by anyone using the existing datasets and scripts.
4. **Benchmark Before Adoption**: No architecture is promoted to Research-OS without objective evidence showing meaningful improvement over the baseline.
5. **Evidence Over Assumptions**: Do not assume that a more complex architecture (e.g., adding an agent, reranker, or knowledge graph) is better. Prove it.

## How to Conduct an Experiment

1. Copy the current Reference Architecture into a new experiment directory (e.g., `A/A4`).
2. Modify exactly one architectural component.
3. Write an implementation for the new component within `core/`.
4. Run the benchmark using the `evaluation/run.py` (or your adapted script).
5. Generate the comparison report using `evaluation/compare.py`.
6. Add a qualitative "Engineering Review" to your generated report.
7. Submit a Pull Request with the benchmark evidence.

## Submitting a Pull Request

- Do NOT delete the previous baseline.
- Do NOT alter `core/` files in a way that breaks previous experiments.
- Always include your `summary.json` and cached metric results so reviewers can examine the data.
