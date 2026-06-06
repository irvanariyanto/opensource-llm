"""Chroma vector store adapter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chromadb

from chunker.schema import Chunk, ChunkMetadata
from vector_store.base_vector_store import BaseVectorStore, SearchResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Vector store adapter backed by ChromaDB.

    Args:
        persist_dir: Directory to persist the Chroma database.
        collection_name: Name of the Chroma collection.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "default",
    ) -> None:
        self._persist_dir = persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore initialised at '%s' (collection: '%s')",
            persist_dir,
            collection_name,
        )

    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Store chunks with their precomputed embeddings in Chroma.

        Args:
            chunks: The document chunks to store.
            embeddings: Embedding vectors aligned 1:1 with chunks.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata.to_dict() for c in chunks],
        )
        logger.info("Stored %d chunks in Chroma", len(chunks))

    def search(
        self, query_embedding: list[float], k: int = 3
    ) -> list[SearchResult]:
        """Search for the most similar chunks.

        Args:
            query_embedding: The query embedding vector.
            k: Number of results to return.

        Returns:
            A list of SearchResult ordered by descending similarity.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunk = Chunk(
                content=doc or "",
                metadata=ChunkMetadata(
                    source=meta.get("source") if meta else None,
                    page=meta.get("page") if meta else None,
                    section=meta.get("section") if meta else None,
                    chunk_index=meta.get("chunk_index", 0) if meta else 0,
                ),
                id=meta.get("id", "") if meta else "",
            )
            # Chroma returns distances (lower = more similar for cosine).
            # Convert to a similarity score.
            score = 1.0 - dist
            search_results.append(SearchResult(chunk=chunk, score=score))

        return search_results
