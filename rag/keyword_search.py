"""BM25 keyword search over ingested chunks.

Dense embeddings alone sometimes miss exact labels and figures ("iPhone",
"$40,665", "Greater China") because semantically similar text can outrank an
exact match. BM25 catches those. The corpus here is small (a few dozen chunks
from a single document), so rebuilding the index per query is cheap and
avoids maintaining a persistent index file — not a choice that would hold up
at a larger document scale, but a reasonable trade-off here.
"""

import re

from rank_bm25 import BM25Okapi

from ingestion.models import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def search(query: str, top_k: int = 6):
    """Return a list of (chroma_id, score) for the top_k BM25 matches.

    Keyed by chroma_id (rather than the Django pk) so results can be merged
    directly with the vector store's hits in rag.retrieval.
    """
    chunks = list(Chunk.objects.values_list("chroma_id", "content_text"))
    if not chunks:
        return []

    ids = [c[0] for c in chunks]
    corpus = [_tokenize(c[1]) for c in chunks]
    bm25 = BM25Okapi(corpus)

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)
    return [(chroma_id, score) for chroma_id, score in ranked[:top_k] if score > 0]
