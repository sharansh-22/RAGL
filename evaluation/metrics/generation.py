"""
RAGL — Generation Metrics
==========================
Objective evaluation of generated answers using NLI and Cross-Encoder models.

Metrics:
  - Faithfulness (NLI): Does the answer only use information from the context?
  - Groundedness (NLI): Are individual claims grounded in the context?
  - Relevancy (Cross-Encoder): Does the answer address the question?

Future metrics (not implemented in A0):
  - Completeness: TODO — future evaluator metric

All evaluations use sentence-transformers CrossEncoder models.
No LLM-as-judge. Fully objective and reproducible.
"""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Loading (cached singletons)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_nli_model(model_name: str) -> CrossEncoder:
    """Load NLI cross-encoder model (cached)."""
    logger.info(f"Loading NLI model on CPU: {model_name}")
    return CrossEncoder(model_name, device="cpu")


@lru_cache(maxsize=1)
def _load_relevancy_model(model_name: str) -> CrossEncoder:
    """Load relevancy cross-encoder model (cached)."""
    logger.info(f"Loading relevancy model on CPU: {model_name}")
    return CrossEncoder(model_name, device="cpu")


# ---------------------------------------------------------------------------
# Sentence Splitting
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for claim-level evaluation."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# Faithfulness (NLI)
# ---------------------------------------------------------------------------

def faithfulness(
    answer: str,
    context: str,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> float:
    """
    Faithfulness via NLI: checks if the answer is entailed by the context.

    Splits the answer into sentences, then for each sentence checks if
    the context entails it. The score is the fraction of answer sentences
    that are entailed (not contradicted) by the context.

    NLI labels: 0=contradiction, 1=entailment, 2=neutral

    Args:
        answer: The generated answer.
        context: The concatenated context from retrieved chunks.
        nli_model_name: Sentence-transformers NLI model.

    Returns:
        Float in [0.0, 1.0]. Higher = more faithful.
    """
    model = _load_nli_model(nli_model_name)
    sentences = _split_sentences(answer)

    if not sentences:
        return 0.0

    # Create pairs: (context, answer_sentence)
    pairs = [(context, sentence) for sentence in sentences]
    scores = model.predict(pairs)

    # For NLI models: scores is array of shape (n, 3) -> [contradiction, entailment, neutral]
    # Or for some models, just logits that need argmax
    if isinstance(scores, np.ndarray) and scores.ndim == 2:
        # Multi-class output: [contradiction, entailment, neutral]
        predictions = np.argmax(scores, axis=1)
        # Count entailment (1) and neutral (2) as non-contradicted
        faithful_count = np.sum(predictions != 0)  # not contradiction
    else:
        # Single score output: higher = more entailed
        faithful_count = np.sum(np.array(scores) > 0.0)

    return float(faithful_count / len(sentences))


# ---------------------------------------------------------------------------
# Groundedness (NLI)
# ---------------------------------------------------------------------------

def groundedness(
    answer: str,
    context: str | list[str],
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> float:
    """
    Groundedness via NLI: checks if each claim in the answer is grounded
    (entailed, not merely neutral) in the context.

    Stricter than faithfulness: requires entailment, not just non-contradiction.
    To avoid truncation issues with long contexts, each sentence is evaluated
    against each chunk individually. A sentence is grounded if at least one
    chunk entails it.

    Args:
        answer: The generated answer.
        context: A string (legacy) or a list of chunk strings.
        nli_model_name: Sentence-transformers NLI model.

    Returns:
        Float in [0.0, 1.0]. Higher = more grounded.
    """
    model = _load_nli_model(nli_model_name)
    sentences = _split_sentences(answer)

    if not sentences:
        return 0.0

    chunks = [context] if isinstance(context, str) else context
    if not chunks:
        return 0.0

    grounded_count = 0

    # Evaluate each sentence individually against the chunks
    for sentence in sentences:
        pairs = [(chunk, sentence) for chunk in chunks]
        scores = model.predict(pairs)
        
        is_grounded = False
        if isinstance(scores, np.ndarray) and scores.ndim == 2:
            # Multi-class: 1 = entailment
            predictions = np.argmax(scores, axis=1)
            if np.any(predictions == 1):
                is_grounded = True
        else:
            if np.any(np.array(scores) > 0.5):
                is_grounded = True
                
        if is_grounded:
            grounded_count += 1

    return float(grounded_count / len(sentences))



# ---------------------------------------------------------------------------
# Relevancy (Cross-Encoder)
# ---------------------------------------------------------------------------

def relevancy(
    answer: str,
    query: str,
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> float:
    """
    Relevancy via Cross-Encoder: does the answer address the question?

    Uses a cross-encoder trained on query-passage relevance to score
    how relevant the answer is to the original query.

    The raw score is passed through a sigmoid to normalize to [0, 1].

    Args:
        answer: The generated answer.
        query: The original question.
        ce_model_name: Sentence-transformers cross-encoder model.

    Returns:
        Float in [0.0, 1.0]. Higher = more relevant.
    """
    model = _load_relevancy_model(ce_model_name)
    score = model.predict([(query, answer)])[0]

    # Normalize with sigmoid
    normalized = 1.0 / (1.0 + np.exp(-float(score)))
    return float(normalized)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_generation_metrics(
    answer: str,
    query: str,
    context: str | list[dict],
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict:
    """
    Compute all generation metrics for a single query-answer pair.

    Returns:
        Dict with keys: faithfulness, groundedness, relevancy.
        Completeness is documented as a future metric (TODO).
    """
    context_text = context if isinstance(context, str) else " ".join(
        c.get("text", "") for c in context
    )
    
    context_chunks = context if isinstance(context, str) else [
        c.get("text", "") for c in context
    ]

    return {
        "faithfulness": faithfulness(answer, context_text, nli_model_name),
        "groundedness": groundedness(answer, context_chunks, nli_model_name),
        "relevancy": relevancy(answer, query, ce_model_name),
        # TODO: Completeness — future evaluator metric.
        # Will require a more sophisticated evaluation approach.
    }
