from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """An in-memory vector store for text chunks."""

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized record, retaining the source document ID."""
        metadata = dict(doc.metadata)
        # Chunk IDs commonly use ``file_id#chunk_number``.  Keep the source
        # ID in metadata so deletion removes every chunk together.
        metadata.setdefault("doc_id", doc.id.split("#", 1)[0])
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": list(self._embedding_fn(doc.content)),
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Search an arbitrary candidate set using dot-product similarity."""
        if top_k <= 0 or not records:
            return []

        query_embedding = self._embedding_fn(query)
        scored = [
            (_dot(query_embedding, record["embedding"]), record)
            for record in records
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        # Embeddings are an internal implementation detail and can be large.
        return [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score,
            }
            for score, record in scored[:top_k]
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document and append its record to the in-memory store."""
        self._store.extend(self._make_record(doc) for doc in docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k records by dot-product similarity."""
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(
        self, query: str, top_k: int = 3, metadata_filter: dict = None
    ) -> list[dict]:
        """Filter metadata first, then perform similarity search."""
        if not metadata_filter:
            candidates = self._store
        else:
            candidates = [
                record
                for record in self._store
                if all(
                    record["metadata"].get(key) == value
                    for key, value in metadata_filter.items()
                )
            ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """Remove all records whose metadata ``doc_id`` matches."""
        original_size = len(self._store)
        self._store = [
            record
            for record in self._store
            if record["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) != original_size
