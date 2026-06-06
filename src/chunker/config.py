"""Configuration for document chunking strategies."""

from dataclasses import dataclass, field
from typing import Literal

_DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class ChunkConfig:
    """Immutable configuration for a text chunker.

    Attributes:
        chunk_size: Maximum number of units (chars or tokens) per chunk.
        chunk_overlap: Number of overlapping units between consecutive chunks.
        separators: Ordered list of separator strings to split on.
        length_function: Unit of measurement — ``"char"`` or ``"token"``.
    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list[str] = field(
        default_factory=lambda: list(_DEFAULT_SEPARATORS),
    )
    length_function: Literal["char", "token"] = "char"

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
