"""Text embedding via the Hugging Face Inference API."""

import os

from huggingface_hub import InferenceClient

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = InferenceClient(api_key=os.environ["HUGGINGFACE_API_KEY"])
    return _client


def embed_text(text: str) -> list[float]:
    """Return an embedding vector for a single text input."""
    output = _get_client().feature_extraction(text, model=EMBEDDING_MODEL)
    if hasattr(output, "tolist"):
        output = output.tolist()
    if isinstance(output, list) and output and isinstance(output[0], list):
        return output[0]
    return output


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for multiple texts (one request per text)."""
    return [embed_text(text) for text in texts]
