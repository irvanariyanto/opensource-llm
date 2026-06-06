"""Base abstraction for document chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loader.base_loader import LoadedDocument
    from chunker.schema import Chunk


class BaseChunker(ABC):
    """Abstract base for all document chunking strategies."""

    @abstractmethod
    def chunk(self, document: LoadedDocument) -> list[Chunk]:
        """Split a loaded document into smaller chunks.

        Args:
            document: A LoadedDocument produced by a loader.

        Returns:
            An ordered list of Chunk objects.
        """
        ...
