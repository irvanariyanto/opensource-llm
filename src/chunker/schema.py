"""Domain types for chunked document segments."""

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class ChunkMetadata:
    """Metadata describing the origin and position of a chunk."""

    source: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict of primitives (for Chroma metadata)."""
        d: dict[str, Any] = {}
        if self.source is not None:
            d["source"] = self.source
        if self.page is not None:
            d["page"] = self.page
        if self.section is not None:
            d["section"] = self.section
        d["chunk_index"] = self.chunk_index
        d.update(self.extra)
        return d


@dataclass
class Chunk:
    """A single chunk of text extracted from a document."""

    content: str
    metadata: ChunkMetadata
    start_index: int = 0
    end_index: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk to a plain dictionary."""
        return asdict(self)
