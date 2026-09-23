"""ChromaDB Cloud wrapper for storing and querying chunk embeddings."""

import os

import chromadb

COLLECTION_NAME = "smart_data_docs"

_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.CloudClient(
            api_key=os.environ["CHROMADB_API_KEY"],
            tenant=os.environ["CHROMADB_TENANT"],
            database=os.environ["CHROMADB_DATABASE"],
        )
    return _client


def get_collection():
    return get_client().get_or_create_collection(COLLECTION_NAME)


def upsert_chunks(ids, embeddings, documents, metadatas):
    get_collection().upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query(embedding, n_results=6):
    return get_collection().query(query_embeddings=[embedding], n_results=n_results)


def reset_collection():
    """Delete and recreate the collection. Used before a fresh ingestion run."""
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(COLLECTION_NAME)
