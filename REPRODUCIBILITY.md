# Reproducibility

To ensure maximum scientific rigor, all benchmark caching and experimental data generated in RAGL v1.0 was produced under a strict execution environment. 

If you are attempting to replicate the evaluations locally, please refer to the following specifications. Differences in CUDA versions, ONNX runtime binaries, and Ollama quantizations will introduce minor floating-point disparities in generation outputs.

## Hardware Specifications
- **CPU**: (Standard x86_64 Host)
- **GPU Accelerator**: NVIDIA / CUDA 12 Compatible

## OS and Environment
- **Operating System**: Linux 
- **Python Version**: 3.10+
- **Environment Manager**: Conda (`environment.yml`)

## Core Dependencies
All dependencies are pinned precisely in `requirements.txt`.
Key versions:
- `onnxruntime-gpu==1.17.1` (Compiled for CUDA 12)
- `fastembed==0.2.7`
- `faiss-cpu==1.8.0`
- `torch==2.3.1`
- `transformers==4.42.4`
- `ollama==0.2.1`

## Local LLM Generator
All evaluation metrics for Generation (Faithfulness, Groundedness, Relevancy) are derived from the local execution of Meta's Llama 3 model via Ollama. 
- **Model Signature**: `llama3:8b` (Default 4-bit quantization provided by the standard Ollama registry)

## Execution Instructions
Always use the provided conda environment to prevent package bleed:

```bash
conda env create -f environment.yml
conda activate ragl
pip install -r requirements.txt
```
