"""RAG pipeline orchestrator — interface-only, zero framework dependencies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pipeline.schema import QueryResult, SourceReference

if TYPE_CHECKING:
    from chunker.base_chunker import BaseChunker
    from chunker.schema import Chunk
    from embedder.base_embedder import BaseEmbedder
    from llm.base_llm import BaseLLM
    from loader.base_loader import BaseLoader
    from vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_TEMPLATE = (
    "Use the following context to answer the question. "
    "If you cannot find the answer in the context, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class RAGPipeline:
    """Orchestrates the full Retrieval-Augmented Generation pipeline.

    This class depends **only** on abstract interfaces (ports).
    It never imports LangChain, Chroma, Ollama, or any other concrete
    library — those live exclusively in the adapter implementations.

    Args:
        loader: Document loader implementation.
        chunker: Text chunker implementation.
        embedder: Embedding provider implementation.
        vector_store: Vector storage/retrieval implementation.
        llm: Language model implementation.
        prompt_template: The prompt template with ``{context}`` and
            ``{question}`` placeholders.
    """

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        prompt_template: str = _DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._prompt_template = prompt_template

    # ── Ingestion ────────────────────────────────────────────────────

    def ingest(self, source: str) -> int:
        """Load, chunk, embed, and store a document.

        Args:
            source: The document source path (e.g. a PDF file path).

        Returns:
            The number of chunks stored.
        """
        # 1. Load
        documents = self._loader.load(source)
        logger.info("Loaded %d pages from '%s'", len(documents), source)

        # 2. Chunk
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self._chunker.chunk(doc))
        logger.info("Split into %d chunks", len(all_chunks))

        if not all_chunks:
            logger.warning("No chunks produced from '%s'", source)
            return 0

        # 3. Embed
        texts = [c.content for c in all_chunks]
        embeddings = self._embedder.embed_texts(texts)
        logger.info("Generated %d embeddings", len(embeddings))

        # 4. Store
        self._vector_store.add_chunks(all_chunks, embeddings)
        logger.info("Stored %d chunks in vector store", len(all_chunks))

        return len(all_chunks)

    # ── Query ────────────────────────────────────────────────────────

    def query(self, question: str, k: int = 3) -> QueryResult:
        """Answer a question using retrieval-augmented generation.

        Args:
            question: The user's natural-language question.
            k: Number of context chunks to retrieve.

        Returns:
            A QueryResult with the answer and source references.
        """
        # 1. Embed the question
        query_embedding = self._embedder.embed_text(question)

        # 2. Retrieve relevant chunks
        search_results = self._vector_store.search(query_embedding, k=k)
        logger.info(
            "Retrieved %d chunks (top score: %.3f)",
            len(search_results),
            search_results[0].score if search_results else 0.0,
        )

        # 3. Build the prompt
        context = "\n\n---\n\n".join(
            sr.chunk.content for sr in search_results
        )
        prompt = self._prompt_template.format(
            context=context, question=question
        )

        # 4. Generate the answer
        answer = self._llm.generate(prompt)

        # 5. Build source references
        sources = [
            SourceReference(
                page=sr.chunk.metadata.page,
                snippet=sr.chunk.content[:150] + "..."
                if len(sr.chunk.content) > 150
                else sr.chunk.content,
                score=sr.score,
            )
            for sr in search_results
        ]

        return QueryResult(answer=answer, sources=sources)
