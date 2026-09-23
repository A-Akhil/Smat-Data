from django.db import models


class Document(models.Model):
    title = models.CharField(max_length=255)
    source_path = models.CharField(max_length=500)
    page_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Chunk(models.Model):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CHUNK_TYPES = [(TEXT, "text"), (TABLE, "table"), (FIGURE, "figure")]

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_type = models.CharField(max_length=10, choices=CHUNK_TYPES)
    page_number = models.IntegerField()
    title = models.CharField(max_length=500, blank=True, default="")

    # Natural-language form: what gets embedded and what the keyword index scores.
    content_text = models.TextField()
    # Structured form (e.g. a Markdown table): passed to the LLM at generation
    # time for tables, where the exact grid matters more than the prose summary.
    content_markdown = models.TextField(blank=True, default="")

    chroma_id = models.CharField(max_length=64, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["document", "chunk_type"])]

    def __str__(self):
        return f"[{self.chunk_type} p{self.page_number}] {self.title or self.content_text[:40]}"
