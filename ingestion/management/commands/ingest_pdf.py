import uuid

from django.conf import settings
from django.core.management.base import BaseCommand

from ingestion.chunker import build_chunks
from ingestion.embeddings import embed_text
from ingestion.models import Chunk, Document
from ingestion.parser import parse_pdf
from ingestion.vectorstore import reset_collection, upsert_chunks


class Command(BaseCommand):
    help = "Parse, chunk, embed, and index a PDF for the RAG pipeline."

    def add_arguments(self, parser):
        parser.add_argument(
            "pdf_path",
            nargs="?",
            default=None,
            help="Path to the PDF to ingest. Defaults to settings.RAG_SOURCE_PDF.",
        )
        parser.add_argument("--title", default=None)

    def handle(self, *args, **options):
        path = options["pdf_path"] or str(settings.RAG_SOURCE_PDF)
        title = options["title"] or path.split("/")[-1]

        self.stdout.write(f"Parsing {path} ...")
        pages = parse_pdf(path)
        chunks = build_chunks(pages)
        self.stdout.write(f"Parsed {len(pages)} pages into {len(chunks)} chunks.")

        # Fresh start: drop any previous ingestion of this document (both the
        # SQL rows and the vector store collection) so re-running is idempotent.
        Document.objects.filter(source_path=path).delete()
        reset_collection()

        doc = Document.objects.create(title=title, source_path=path, page_count=len(pages))

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks, start=1):
            chroma_id = str(uuid.uuid4())
            embedding = embed_text(chunk["content_text"])

            Chunk.objects.create(
                document=doc,
                chunk_type=chunk["chunk_type"],
                page_number=chunk["page_number"],
                title=chunk["title"],
                content_text=chunk["content_text"],
                content_markdown=chunk["content_markdown"],
                chroma_id=chroma_id,
            )
            ids.append(chroma_id)
            embeddings.append(embedding)
            documents.append(chunk["content_text"])
            metadatas.append(
                {
                    "page": chunk["page_number"],
                    "type": chunk["chunk_type"],
                    "title": chunk["title"],
                }
            )
            self.stdout.write(f"  embedded chunk {i}/{len(chunks)}", ending="\r")

        upsert_chunks(ids, embeddings, documents, metadatas)
        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(f"Ingested {len(chunks)} chunks from '{title}' into ChromaDB.")
        )
