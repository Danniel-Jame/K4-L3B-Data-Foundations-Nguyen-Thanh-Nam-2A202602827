from __future__ import annotations

import re
from pathlib import Path

from src.chunking import (
    ChunkingStrategyComparator,
    SemanticSimilarityChunker,
)
from src.embeddings import (
    LOCAL_EMBEDDING_MODEL,
    LocalEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


CORPUS_DIR = Path("data/shopee-return-refund")
SAMPLE_DOCS = [
    CORPUS_DIR / "buyer-return-request.md",
    CORPUS_DIR / "buyer-refund-processing.md",
    CORPUS_DIR / "seller-return-handling.md",
]

QUERIES = [
    ("Buyer return window", "How long does a buyer have to request a return or refund?", {"audience": "buyer"}),
    ("Seller response duties", "What must a seller do after a buyer opens a return request?", {"audience": "seller"}),
    ("Refund timing", "When is the buyer's refund released after a return is approved?", {"audience": "buyer"}),
    ("Eligible reasons", "Which reasons and evidence can support a return or refund claim?", None),
    ("Seller shipping", "Who pays return shipping and what should the seller do with the parcel?", {"audience": "seller"}),
]


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
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
            metadata[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return metadata, "\n".join(lines[end + 1 :]).strip()


def select_embedder():
    try:
        return LocalEmbedder(model_name=LOCAL_EMBEDDING_MODEL), "local multilingual"
    except Exception:
        return _mock_embed, "mock fallback (not semantic)"


def load_corpus(chunker: SemanticSimilarityChunker) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        metadata, content = parse_frontmatter(path.read_text(encoding="utf-8"))
        if not metadata:
            raise ValueError(f"Missing frontmatter: {path}")
        for index, chunk in enumerate(chunker.chunk(content)):
            chunk_metadata = dict(metadata)
            chunk_metadata["source"] = str(path).replace("\\", "/")
            documents.append(Document(id=f"{metadata['doc_id']}#{index}", content=chunk, metadata=chunk_metadata))
    return documents


def stats(chunks: list[str]) -> tuple[int, float]:
    return len(chunks), (sum(map(len, chunks)) / len(chunks) if chunks else 0.0)


def main() -> None:
    embedder, backend = select_embedder()
    semantic = SemanticSimilarityChunker(
        embedding_fn=embedder,
        similarity_threshold=0.35,
        max_chunk_size=420,
    )

    print("=== Semantic Similarity Chunking ===")
    print(f"Embedding backend: {backend}")
    print("Parameters: threshold=0.35, max_chunk_size=420")
    print("\n=== Baseline vs semantic statistics ===")
    for path in SAMPLE_DOCS:
        _, content = parse_frontmatter(path.read_text(encoding="utf-8"))
        baseline = ChunkingStrategyComparator().compare(content, chunk_size=420)
        semantic_stats = stats(semantic.chunk(content))
        print(f"{path.name}:")
        for name in ("fixed_size", "by_sentences", "recursive"):
            print(f"  {name}: count={baseline[name]['count']}, avg_length={baseline[name]['avg_length']:.1f}")
        print(f"  semantic_similarity: count={semantic_stats[0]}, avg_length={semantic_stats[1]:.1f}")

    documents = load_corpus(semantic)
    store = EmbeddingStore(collection_name="semantic-similarity", embedding_fn=embedder)
    store.add_documents(documents)
    print(f"\nCorpus documents: {len({doc.metadata['doc_id'] for doc in documents})}")
    print(f"Indexed chunks: {store.get_collection_size()}")
    for number, (label, query, metadata_filter) in enumerate(QUERIES, start=1):
        results = (
            store.search_with_filter(query, top_k=3, metadata_filter=metadata_filter)
            if metadata_filter
            else store.search(query, top_k=3)
        )
        print(f"\n{number}. {label} | filter={metadata_filter or 'none'}")
        for rank, result in enumerate(results, start=1):
            print(
                f"  {rank}. score={result['score']:.6f} "
                f"source_id={result['metadata'].get('doc_id', result['id'])} "
                f"chunk_id={result['id']}"
            )


if __name__ == "__main__":
    main()
