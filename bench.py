from __future__ import annotations

import re
from pathlib import Path

from src.chunking import SemanticSimilarityChunker
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore


CORPUS_DIR = Path("data/shopee-return-refund")

QUERIES = [
    ("Buyer return window", "How long does a buyer have to request a return or refund?", {"audience": "buyer"}),
    ("Seller response duties", "What must a seller do after a buyer opens a return request?", {"audience": "seller"}),
    ("Refund timing", "When is the buyer's refund released after a return is approved?", {"audience": "buyer"}),
    ("Eligible reasons", "Which reasons and evidence can support a return or refund claim?", None),
    ("Seller shipping", "Who pays return shipping and what should the seller do with the parcel?", {"audience": "seller"}),
]


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse the simple key/value YAML frontmatter used by the corpus."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, text
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if match:
            value = match.group(2).strip().strip('"').strip("'")
            metadata[match.group(1)] = value
    return metadata, "\n".join(lines[end + 1 :]).strip()


def load_corpus() -> list[Document]:
    documents: list[Document] = []
    chunker = SemanticSimilarityChunker(
        embedding_fn=_mock_embed,
        similarity_threshold=0.35,
        max_chunk_size=420,
    )
    for path in sorted(CORPUS_DIR.glob("*.md")):
        metadata, content = parse_frontmatter(path.read_text(encoding="utf-8"))
        if not metadata:
            raise ValueError(f"Missing frontmatter: {path}")
        for index, chunk in enumerate(chunker.chunk(content)):
            chunk_metadata = dict(metadata)
            chunk_metadata["source"] = str(path).replace("\\", "/")
            documents.append(
                Document(
                    id=f"{metadata['doc_id']}#{index}",
                    content=chunk,
                    metadata=chunk_metadata,
                )
            )
    return documents


def main() -> None:
    documents = load_corpus()
    store = EmbeddingStore(collection_name="shopee-return-refund", embedding_fn=_mock_embed)
    store.add_documents(documents)
    print(f"Corpus documents: {len({doc.metadata['doc_id'] for doc in documents})}")
    print(f"Indexed chunks: {store.get_collection_size()}")
    for number, (label, query, metadata_filter) in enumerate(QUERIES, start=1):
        results = (
            store.search_with_filter(query, top_k=3, metadata_filter=metadata_filter)
            if metadata_filter
            else store.search(query, top_k=3)
        )
        filter_text = repr(metadata_filter) if metadata_filter else "none"
        print(f"\n{number}. {label}")
        print(f"Query: {query}")
        print(f"Metadata filter: {filter_text}")
        for rank, result in enumerate(results, start=1):
            source_id = result["metadata"].get("doc_id", result["id"])
            print(f"  {rank}. score={result['score']:.6f} source_id={source_id} chunk_id={result['id']}")


if __name__ == "__main__":
    main()