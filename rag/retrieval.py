"""Hybrid retrieval: merges dense (ChromaDB) and keyword (BM25) search results."""

from ingestion.embeddings import embed_text
from ingestion.models import Chunk
from ingestion.vectorstore import query as vector_query
from rag.keyword_search import search as bm25_search


def hybrid_search(question: str, top_k: int = 6):
    """Return the top_k Chunk objects most relevant to the question, ranked
    by a simple weighted merge of dense-similarity and BM25 scores.

    Score normalization here is deliberately simple (min-max per source) —
    good enough at this document's scale; a larger corpus would warrant a
    proper reranker instead of ad-hoc score fusion.
    """
    scores: dict[str, float] = {}

    dense = vector_query(embed_text(question), n_results=top_k)
    dense_ids = dense["ids"][0] if dense["ids"] else []
    dense_distances = dense["distances"][0] if dense["distances"] else []
    if dense_distances:
        max_dist = max(dense_distances) or 1.0
        for chroma_id, distance in zip(dense_ids, dense_distances):
            similarity = 1 - (distance / max_dist)
            scores[chroma_id] = scores.get(chroma_id, 0) + 0.6 * similarity

    bm25_hits = bm25_search(question, top_k=top_k)
    if bm25_hits:
        max_score = max(score for _, score in bm25_hits) or 1.0
        for chroma_id, score in bm25_hits:
            scores[chroma_id] = scores.get(chroma_id, 0) + 0.4 * (score / max_score)

    ranked_ids = [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)][:top_k]
    if not ranked_ids:
        return []

    chunks = {c.chroma_id: c for c in Chunk.objects.filter(chroma_id__in=ranked_ids)}
    return [chunks[cid] for cid in ranked_ids if cid in chunks]
