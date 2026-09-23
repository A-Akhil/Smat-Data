#!/bin/sh
set -e

# The source PDF is gitignored (see README), so a fresh deploy needs to fetch
# it before ingesting. RAG_SOURCE_PDF_URL defaults to the assignment's own
# test-link PDF; override it to deploy against a different document.
PDF_PATH="${RAG_SOURCE_PDF_PATH:-/app/2022_Q3_AAPL.pdf}"
PDF_URL="${RAG_SOURCE_PDF_URL:-https://github.com/docugami/KG-RAG-datasets/raw/main/sec-10-q/data/v1/docs/2022%20Q3%20AAPL.pdf}"

if [ ! -f "$PDF_PATH" ]; then
    echo "Fetching source PDF from $PDF_URL ..."
    curl -sL -o "$PDF_PATH" "$PDF_URL"
fi

python manage.py migrate --noinput
python manage.py seed_api_keys

# Always re-ingest on startup: ingest_pdf wipes and rebuilds both the SQLite
# rows and the ChromaDB collection together (see ingestion/management/
# commands/ingest_pdf.py), which keeps the two in sync even if the
# container's local disk doesn't persist across deploys/restarts. Costs a
# short cold-start (re-embedding ~60 chunks) — an acceptable trade-off for a
# single small document; a larger corpus would want to skip this when
# nothing has changed instead.
python manage.py ingest_pdf

exec gunicorn core.wsgi:application --bind "0.0.0.0:${PORT:-8000}"
