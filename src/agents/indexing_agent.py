import hashlib
import logging

import chromadb

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
MIN_CHUNK_CHARS = 50


class IndexingAgent:
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0

        ids, documents, metadatas = [], [], []

        for chunk in chunks:
            text = chunk["text"]
            meta = chunk["metadata"]

            if len(text.strip()) < MIN_CHUNK_CHARS:
                continue

            chunk_id = hashlib.sha256(
                f"{meta.get('source_file', '')}::{meta.get('chunk_index', 0)}".encode()
            ).hexdigest()[:32]

            # chroma only accepts str/int/float/bool in metadata
            safe_meta = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v) for k, v in meta.items()}

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(safe_meta)

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("indexed %d chunks", len(ids))
        return len(ids)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]

    def collection_size(self) -> int:
        return self.collection.count()
