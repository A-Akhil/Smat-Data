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
    """Render every row as a table body row.

    grid[0] is NOT a real header — these filings' actual period headers have
    no ruling lines, so pdfplumber's table detector already excludes them
    from the grid entirely (recovered separately as header_preamble, see
    parser.py). grid[0] is just the first data row (sometimes a value row,
    sometimes a blank section label like "ASSETS:"); treating it as a header
    would both drop it as data and falsely imply it names the columns.
    """
    if not grid:
        return ""
    num_cols = max(len(row) for row in grid)
    lines = [
        "| " + " | ".join([""] * num_cols) + " |",
        "| " + " | ".join(["---"] * num_cols) + " |",
    ]
    for row in grid:
        row = row + [""] * (num_cols - len(row))
        lines.append("| " + " | ".join(row[:num_cols]) + " |")
    return "\n".join(lines)


def _grid_to_sentences(grid, title):
    """One sentence per row: '<label> — <value>, <value>, ...'.

    No header row to pair against (see _grid_to_markdown) — the recovered
    header_preamble text, prepended once for the whole table by the caller,
    carries the column meaning instead. Lone '$' cells (a currency symbol
    split into its own grid cell) are dropped as noise.
    """
    prefix = f"{title}: " if title else ""
    sentences = []
    for row in grid:
        if not row or not row[0]:
            continue
        label = row[0]
        values = [cell for cell in row[1:] if cell and cell != "$"]
        if values:
            sentences.append(f"{prefix}{label} — " + ", ".join(values))
        else:
            # Section label with no values on its own row (e.g. "Cost of sales:").
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
