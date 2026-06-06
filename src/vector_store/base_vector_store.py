"""Base abstraction for vector storage and retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chunker.schema import Chunk


@dataclass(frozen=True)
class SearchResult:
    """A single result from a vector similarity search.

    Attributes:
        chunk: The matching chunk.
        score: Similarity score (higher is more similar).
    """

    chunk: Chunk
    score: float


class BaseVectorStore(ABC):
    """Abstract base for all vector store backends.

    Implementations handle storing chunk embeddings and performing
    similarity searches against them.
    """

    @abstractmethod
    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Store chunks with their precomputed embedding vectors.

        Args:
            chunks: The document chunks to store.
            embeddings: The embedding vectors, one per chunk.
                Must have the same length as ``chunks``.
        """
        ...

    @abstractmethod
    def search(
        self, query_embedding: list[float], k: int = 3
    ) -> list[SearchResult]:
        """Find the most similar chunks to a query embedding.

        Args:
            query_embedding: The embedding vector of the query.
            k: Number of results to return.

        Returns:
            A list of SearchResult ordered by descending similarity.
        """
        ...
