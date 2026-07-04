"""
RAGL — Generator
=================
Responsible for:
  - Constructing a RAG prompt from query + context chunks
  - Querying llama3:8b via Ollama
  - Returning the generated answer

Not responsible for retrieval, embedding, or indexing.
"""

import logging

import ollama

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed Prompt Template (A0 Reference Implementation)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a knowledgeable research assistant. Answer the user's question using ONLY the provided context. If the context does not contain sufficient information to answer the question, say so clearly. Do not make up information.

Rules:
1. Base your answer strictly on the provided context.
2. If the answer requires information not in the context, state that explicitly.
3. Be precise and concise.
4. When relevant, reference the source document."""

_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}

Answer:"""


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build the user prompt from a query and retrieved context chunks.

    Args:
        query: The user's question.
        context_chunks: List of chunk dicts with 'text' and 'source' fields.

    Returns:
        The formatted user prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[{i}] (Source: {source})\n{text}")

    context = "\n\n".join(context_parts)
    return _USER_PROMPT_TEMPLATE.format(context=context, query=query)


# ---------------------------------------------------------------------------
# LLM Generation
# ---------------------------------------------------------------------------

def generate(prompt: str, model_name: str = "llama3:8b") -> str:
    """
    Generate an answer using Ollama.

    Args:
        prompt: The formatted user prompt (from build_prompt).
        model_name: Ollama model identifier.

    Returns:
        The generated answer string.
    """
    logger.debug(f"Generating with model: {model_name}")

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response["message"]["content"]
    logger.debug(f"Generated {len(answer)} characters")
    return answer
