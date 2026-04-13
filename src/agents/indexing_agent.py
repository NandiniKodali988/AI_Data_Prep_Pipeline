import hashlib
import logging

import chromadb
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
# skip anything shorter than this — usually stray headers or page numbers
MIN_CHUNK_CHARS = 50
# RRF constant — 60 is the standard value from the original paper
_RRF_K = 60


class IndexingAgent:
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_docs = []  # list of (chunk_id, text, metadata)
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """Build BM25 index from all documents currently in ChromaDB."""
        results = self.collection.get(include=["documents", "metadatas"])
        if not results["ids"]:
            self._bm25 = None
            self._bm25_docs = []
            return
        self._bm25_docs = list(zip(results["ids"], results["documents"], results["metadatas"]))
        tokenized = [doc.lower().split() for _, doc, _ in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug("BM25 index built with %d documents", len(self._bm25_docs))

    def index(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0

        ids, documents, metadatas = [], [], []

        for chunk in chunks:
            text = chunk["text"]
            meta = chunk["metadata"]

            if len(text.strip()) < MIN_CHUNK_CHARS:
                continue

            # deterministic ID so re-indexing the same file is a no-op (upsert)
            chunk_id = hashlib.sha256(
                f"{meta.get('source_file', '')}::{meta.get('chunk_index', 0)}".encode()
            ).hexdigest()[:32]

            # chroma only accepts str/int/float/bool in metadata
            safe_meta = {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in meta.items()
            }

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(safe_meta)

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        # rebuild BM25 so new documents are immediately searchable
        self._rebuild_bm25()
        logger.info("indexed %d chunks", len(ids))
        return len(ids)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        total = self.collection.count()
        if total == 0:
            return []

        # retrieve more candidates from each source before fusion
        candidate_k = min(top_k * 3, total)

        # --- semantic search ---
        sem_results = self.collection.query(
            query_texts=[query],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        sem_ids = sem_results["ids"][0]

        # --- BM25 keyword search ---
        bm25_ids = []
        if self._bm25 and self._bm25_docs:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
            # only include results with a non-zero BM25 score
            bm25_ids = [self._bm25_docs[i][0] for i, score in ranked[:candidate_k] if score > 0]

        # --- Reciprocal Rank Fusion ---
        # each list contributes 1/(k + rank) to a document's score
        rrf_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(sem_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (_RRF_K + rank + 1)
        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (_RRF_K + rank + 1)

        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        # build lookups so we can reconstruct full chunk dicts
        sem_lookup = {
            doc_id: {"text": doc, "metadata": meta, "distance": dist}
            for doc_id, doc, meta, dist in zip(
                sem_results["ids"][0],
                sem_results["documents"][0],
                sem_results["metadatas"][0],
                sem_results["distances"][0],
            )
        }
        bm25_lookup = {
            doc_id: {"text": text, "metadata": meta} for doc_id, text, meta in self._bm25_docs
        }

        max_rrf = max(rrf_scores.values()) if rrf_scores else 1
        chunks = []
        for doc_id in top_ids:
            if doc_id in sem_lookup:
                entry = sem_lookup[doc_id].copy()
            elif doc_id in bm25_lookup:
                entry = {**bm25_lookup[doc_id], "distance": 1.0}
            else:
                continue
            # normalize to [0, 1] so the UI can display a human-readable relevance score
            entry["score"] = rrf_scores[doc_id] / max_rrf
            chunks.append(entry)

        return chunks

    def collection_size(self) -> int:
        return self.collection.count()
