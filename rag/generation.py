"""Builds LLM context from retrieved chunks and generates a cited answer."""

from rag.llm_client import GroqClient
from rag.retrieval import hybrid_search

SYSTEM_PROMPT = (
    "You are a financial document Q&A assistant answering questions about an "
    "Apple Inc. Form 10-Q filing. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say you don't know rather "
    "than guessing. When you use a table, read its column headers carefully "
    "before quoting a number — headers may list multiple time periods side "
    "by side. Always mention the page number(s) your answer is drawn from."
)


def _format_chunk(chunk):
    header = f"[Page {chunk.page_number}, {chunk.chunk_type}]"
    if chunk.title:
        header += f" {chunk.title}"
    body = chunk.content_markdown if chunk.chunk_type == "table" and chunk.content_markdown else chunk.content_text
    return f"{header}\n{body}"


def build_context(chunks) -> str:
    return "\n\n---\n\n".join(_format_chunk(c) for c in chunks)


def answer_question(question: str, top_k: int = 6):
    """Returns (answer_text, source_chunks)."""
    chunks = hybrid_search(question, top_k=top_k)
    if not chunks:
        return "I couldn't find anything relevant to that question in the document.", []

    context = build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    answer = GroqClient().chat(messages)
    return answer, chunks
