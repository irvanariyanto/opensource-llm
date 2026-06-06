"""Base abstraction for document loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentMetadata:
    """Structured metadata attached to a loaded document."""

    source: str
    page: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedDocument:
    """A single document page/section returned by a loader."""

    page_content: str
    metadata: DocumentMetadata


class BaseLoader(ABC):
    """Abstract base for all document loaders."""

    @abstractmethod
    def load(self, source: str) -> list[LoadedDocument]:
        """Load a document from a source and return a list of LoadedDocument."""
        ...

    @staticmethod
    @abstractmethod
    def supports(source: str) -> bool:
        """Check if the loader supports the given source."""
        ...
