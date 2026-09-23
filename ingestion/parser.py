"""Layout-aware PDF parsing.

Plain text extraction flattens tables into unreadable word-soup because it
reads left-to-right, top-to-bottom regardless of columns. This module keeps
tables and narrative text separate: it finds each table's bounding box on the
page, extracts it as a structured grid, and extracts the remaining page
content (everything outside those boxes) as narrative text.
"""

import pymupdf


def _obj_center_in_bbox(obj, bbox):
    x0, top, x1, bottom = bbox
    cx = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
    cy = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def _clean_cell(value):
    if value is None:
        return ""
    return " ".join(value.split())


def parse_pdf(path):
    """Parse a PDF into a per-page structure.

    Returns a list of dicts, one per page:
        {
            "page": <1-indexed page number>,
            "text": "<narrative text with table regions excluded>",
            "tables": [{"grid": [[cell, ...], ...], "title": "<nearest heading above it, or None>"}],
            "images": [<count of embedded raster images on the page>],
        }
    """
    import pdfplumber

    pages_out = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            if table_bboxes:
                narrative_page = page.filter(
                    lambda obj: not any(_obj_center_in_bbox(obj, bbox) for bbox in table_bboxes)
                )
            else:
                narrative_page = page

            narrative_text = (narrative_page.extract_text() or "").strip()

            narrative_lines = []
            try:
                for line in narrative_page.extract_text_lines():
                    narrative_lines.append((line["top"], line["text"].strip()))
            except Exception:
                pass

            page_tables = []
            for table in tables:
                grid = [[_clean_cell(cell) for cell in row] for row in (table.extract() or [])]
                grid = [row for row in grid if any(cell for cell in row)]
                if not grid:
                    continue

                # Column-period headers (e.g. "Three Months Ended ... June 25,
                # 2022 / June 26, 2021 ...") in these filings have no ruling
                # lines, so pdfplumber's line-based table detector excludes
                # them from the grid entirely — the grid starts at the first
                # data row. Recover them as a text preamble by walking
                # narrative lines upward from the table's top edge while the
                # line spacing stays tight (a real paragraph break has a much
                # bigger gap), which reliably captures just the header block.
                lines_above = sorted(
                    ((top, txt) for top, txt in narrative_lines if txt and top < table.bbox[1]),
                    reverse=True,
                )
                header_preamble = []
                cursor = table.bbox[1]
                for top, txt in lines_above:
                    if len(header_preamble) >= 5 or cursor - top > 15:
                        break
                    header_preamble.insert(0, txt)
                    cursor = top

                # Prefer the nearest ALL-CAPS section heading above the table
                # (10-Q headings like "CONDENSED CONSOLIDATED STATEMENTS OF
                # OPERATIONS" are set in caps) over the literal nearest line,
                # which is sometimes just a numeric row that leaked past the
                # table's detected bounding box (e.g. "Net income $ 19,442 ...").
                def _is_label_like(text):
                    digits = sum(c.isdigit() for c in text)
                    return digits < max(3, len(text) // 4)

                above = [
                    (top, text)
                    for (top, text) in narrative_lines
                    if text and top < table.bbox[1] and _is_label_like(text)
                ]
                def _looks_like_heading(text):
                    alpha = [c for c in text if c.isalpha()]
                    if len(alpha) < 8:
                        return False
                    upper_ratio = sum(c.isupper() for c in alpha) / len(alpha)
                    return upper_ratio > 0.8

                title = None
                for top, text in reversed(above):
                    if table.bbox[1] - top > 400:
                        break
                    if _looks_like_heading(text):
                        title = text
                        break
                if title is None:
                    nearby = [text for top, text in above if table.bbox[1] - top < 200]
                    title = nearby[-1] if nearby else None
                page_tables.append(
                    {"grid": grid, "title": title, "header_preamble": header_preamble}
                )

            pages_out.append(
                {
                    "page": page_number,
                    "text": narrative_text,
                    "tables": page_tables,
                    "image_count": len(page.images),
                }
            )

    return pages_out


def extract_images(path, out_dir):
    """Extract embedded raster images from the PDF, saved to out_dir.

    Returns a list of {"page": <1-indexed page number>, "path": <saved file path>}.
    Used defensively: this document has none, but the ingestion pipeline
    should not silently drop figures if a different source PDF has them.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    results = []
    doc = pymupdf.open(path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for image_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                ext = base_image.get("ext", "png")
                out_path = os.path.join(out_dir, f"page{page_index + 1}_img{image_index}.{ext}")
                with open(out_path, "wb") as f:
                    f.write(base_image["image"])
                results.append({"page": page_index + 1, "path": out_path})
    finally:
        doc.close()
    return results
