from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """Retrieve relevant chunks, inject source-aware context, and ask the LLM."""
        if not question or not question.strip():
            return "I couldn't answer because the question was empty."

        if self.store is None or self.store.get_collection_size() == 0:
            return "I couldn't find any relevant information in the knowledge base."

        if top_k <= 0:
            return "I couldn't find any relevant information in the knowledge base."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "I couldn't find any relevant information in the knowledge base."

        context_parts: list[str] = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source = (
                metadata.get("source")
                or metadata.get("source_url")
                or metadata.get("doc_id")
                or result.get("id")
                or f"chunk-{index}"
            )
            metadata_text = ", ".join(
                f"{key}={value}"
                for key, value in sorted(metadata.items())
                if value is not None and key != "embedding"
            )
            if not metadata_text:
                metadata_text = "no metadata"

            content = (result.get("content") or "").strip()
            context_parts.append(
                f"[{index}] Source: {source}\n"
                f"Metadata: {metadata_text}\n"
                f"Content: {content}"
            )

        context_block = "\n\n".join(context_parts)
        prompt = (
            "You are a careful assistant. Answer the user's question using only the context below and the source information provided. "
            "If the context does not contain enough information, say so clearly rather than guessing.\n\n"
            f"Question: {question}\n\n"
            "Relevant context:\n"
            f"{context_block}\n\n"
            "Cite the source numbers [1], [2], etc. in your answer when relevant.\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
