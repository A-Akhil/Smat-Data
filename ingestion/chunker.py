"""Turns parsed pages into retrievable chunks.

Two chunk types, deliberately kept simple given the size of a single 28-page
document (no need for semantic/heading-based sub-chunking at this scale):

- "text": one chunk per page's narrative content.
- "table": one chunk per detected table, kept whole (never split), stored in
  two forms — a natural-language row-by-row description (used for embedding
  and keyword search, since dense/BM25 retrieval work better over prose than
  a raw grid) and a Markdown table (passed to the LLM at generation time,
  since the exact grid is what the model should reason over for numbers).
"""

MIN_TEXT_CHUNK_CHARS = 30


def _grid_to_markdown(grid):
    if not grid:
        return ""
    header, *rows = grid
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        row = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(row[: len(header)]) + " |")
    return "\n".join(lines)


def _grid_to_sentences(grid, title):
    if len(grid) < 2:
        return ""
    header, *rows = grid
    prefix = f"{title}: " if title else ""
    sentences = []
    for row in rows:
        if not row or not row[0]:
            continue
        label = row[0]
        pairs = [
            f"{h.strip()} = {v.strip()}"
            for h, v in zip(header[1:], row[1:])
            if h.strip() and v.strip()
        ]
        if pairs:
            sentences.append(f"{prefix}{label} — " + "; ".join(pairs))
        else:
            # Header/section row with no aligned values (e.g. "Cost of sales:").
            sentences.append(f"{prefix}{label}")
    return "\n".join(sentences)


def build_chunks(pages):
    """Build a flat list of chunk dicts from parser.parse_pdf() output.

    Each dict: {chunk_type, page_number, title, content_text, content_markdown}
    """
    chunks = []

    for page in pages:
        text = page["text"].strip()
        if len(text) >= MIN_TEXT_CHUNK_CHARS:
            chunks.append(
                {
                    "chunk_type": "text",
                    "page_number": page["page"],
                    "title": "",
                    "content_text": text,
                    "content_markdown": "",
                }
            )

        for table in page["tables"]:
            grid = table["grid"]
            title = table["title"] or ""
            preamble = table.get("header_preamble") or []
            preamble_line = " / ".join(preamble)
            sentences = _grid_to_sentences(grid, title)
            markdown = _grid_to_markdown(grid)
            if not sentences and not markdown:
                continue

            header_note = f"Column headers (recovered from surrounding text): {preamble_line}\n" if preamble_line else ""
            content_text = header_note + ((f"{title}\n" if title else "") + sentences if sentences else title)
            content_markdown = (f"*{preamble_line}*\n\n" if preamble_line else "") + markdown

            chunks.append(
                {
                    "chunk_type": "table",
                    "page_number": page["page"],
                    "title": title,
                    "content_text": content_text or markdown,
                    "content_markdown": content_markdown,
                }
            )

        if page.get("image_count"):
            chunks.append(
                {
                    "chunk_type": "figure",
                    "page_number": page["page"],
                    "title": "",
                    "content_text": f"[{page['image_count']} embedded image(s) on page {page['page']}, "
                    "not further analyzed — see design write-up for scope]",
                    "content_markdown": "",
                }
            )

    return chunks
