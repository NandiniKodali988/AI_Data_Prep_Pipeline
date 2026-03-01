"""Indexing Agent — stores document chunks in ChromaDB with embeddings."""
import hashlib
import logging

import chromadb

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"


class IndexingAgent:
    """
    Loads document chunks into a persistent ChromaDB collection.

    ChromaDB's default embedding function (sentence-transformers) is used
    so no API key is needed for embeddings. Chunks are upserted by a
    content-hash ID so re-running the pipeline is idempotent.
    """

    def __init__(self, chroma_db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, chunks: list[dict]) -> int:
        """
        Upsert chunks into ChromaDB.

        Args:
            chunks: List of {"text": str, "metadata": dict} dicts.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        MIN_CHUNK_CHARS = 50

        for chunk in chunks:
            text = chunk["text"]
            meta = chunk["metadata"]

            if len(text.strip()) < MIN_CHUNK_CHARS:
                logger.debug("Skipping short chunk (%d chars): %r", len(text), text[:40])
                continue

            # Stable ID based on source + chunk index
            chunk_id = hashlib.sha256(
                f"{meta.get('source_file', '')}::{meta.get('chunk_index', 0)}".encode()
            ).hexdigest()[:32]

            # ChromaDB metadata values must be str, int, float, or bool
            safe_meta = {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in meta.items()
            }

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(safe_meta)

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Indexed %d chunks into ChromaDB collection '%s'", len(ids), COLLECTION_NAME)
        return len(ids)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search over indexed chunks.

        Returns:
            List of {"text": str, "metadata": dict, "distance": float} dicts.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"text": doc, "metadata": meta, "distance": dist})

        return hits

    def collection_size(self) -> int:
        return self.collection.count()
