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


def _row_lines(content_text: str, title: str) -> list[str]:
    """Split a table chunk's prose form back into its individual row lines.

    A whole-statement table chunk can run to 20+ rows (see
    ingestion/chunker.py — tables are kept whole, never split, so the LLM
    always gets the full table at generation time). But that means a query
    naming one specific row, e.g. "net income", competes for BM25 relevance
    against the other ~19 rows' worth of unrelated tokens in the same
    document, and loses to shorter chunks that merely share boilerplate
    phrasing. Indexing each row as its own small BM25 document (still
    resolving back to the same parent chunk) fixes that without changing
    what's actually retrieved or shown — see search() below.
    """
    prefix = f"{title}: " if title else ""
    lines = []
    for line in content_text.splitlines():
        if line.startswith("Column headers") or line == title:
            continue
        if prefix and line.startswith(prefix):
            line = line[len(prefix) :]
        if line.strip():
            lines.append(line)
    return lines


# Generic English function words plus this document's own boilerplate
# ("three months ended June 25, 2022" appears almost everywhere) — excluded
# only from phrase-bonus n-grams below, not from ordinary BM25 scoring,
# so a bigram bonus reflects an actually distinctive phrase like "net
# income" rather than a boilerplate one like "months ended" that would
# otherwise match nearly every table in the document.
_STOPWORDS = {
    "the", "a", "an", "of", "for", "was", "were", "is", "are", "in", "on", "to",
    "and", "or", "as", "at", "by", "that", "this", "it", "its", "what", "how",
    "did", "do", "does", "much", "many",
    "three", "six", "nine", "twelve", "month", "months", "ended", "quarter",
    "year", "years", "fiscal",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def _ngrams(tokens: list[str], n: int) -> set[tuple]:
    # Alphabetic content words only — a shared numeric bigram like ("25",
    # "2022") from a date is just as generic here as "months ended" and
    # would otherwise reward coincidental date adjacency instead of an
    # actual phrase match like "net income" or "total assets".
    content_tokens = [t for t in tokens if t.isalpha() and t not in _STOPWORDS]
    return {tuple(content_tokens[i : i + n]) for i in range(len(content_tokens) - n + 1)}


def search(query: str, top_k: int = 6):
    """Return a list of (chroma_id, score) for the top_k BM25 matches.

    Keyed by chroma_id (rather than the Django pk) so results can be merged
    directly with the vector store's hits in rag.retrieval. Internally scores
    both whole chunks and, for tables, each individual row line; a chunk's
    final score is the best of any of its own documents.
    """
    chunks = list(Chunk.objects.values_list("chroma_id", "content_text", "title", "chunk_type"))
    if not chunks:
        return []

    doc_chroma_ids = []
    corpus = []
    for chroma_id, content_text, title, chunk_type in chunks:
        doc_chroma_ids.append(chroma_id)
        corpus.append(_tokenize(content_text))

        if chunk_type == "table":
            for line in _row_lines(content_text, title):
                doc_chroma_ids.append(chroma_id)
                corpus.append(_tokenize(line))

    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Plain BM25 struggles here: this document repeats generic boilerplate
    # ("Three Months Ended... June 25, 2022...") in nearly every table, and
    # a term like "net" or "total" is so common (as in "net sales", "total
    # net sales", "total operating expenses") that its IDF is close to zero
    # — so "net income" or "total assets" barely outscores unrelated chunks
    # on unigram overlap alone. An exact-phrase bonus fixes this, weighted by
    # the bigram/trigram's own average IDF rather than a flat amount: a rare,
    # specific phrase match (e.g. "net income", where "income" is
    # meaningfully rare) is rewarded far more than one built from generic
    # words, so this doesn't just reduce to another flat-boilerplate boost.
    query_bigrams = _ngrams(query_tokens, 2)
    query_trigrams = _ngrams(query_tokens, 3)

    def _phrase_weight(ngram, per_token_weight):
        avg_idf = sum(bm25.idf.get(tok, 0) for tok in ngram) / len(ngram)
        return avg_idf * per_token_weight

    for i, doc_tokens in enumerate(corpus):
        for bigram in query_bigrams & _ngrams(doc_tokens, 2):
            scores[i] += _phrase_weight(bigram, 15)
        for trigram in query_trigrams & _ngrams(doc_tokens, 3):
            scores[i] += _phrase_weight(trigram, 15)

    best_score = {}
    for chroma_id, score in zip(doc_chroma_ids, scores):
        if score > best_score.get(chroma_id, 0):
            best_score[chroma_id] = score

    ranked = sorted(best_score.items(), key=lambda pair: pair[1], reverse=True)
    return [(chroma_id, score) for chroma_id, score in ranked[:top_k] if score > 0]
