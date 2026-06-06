"""Document chunking abstractions, configuration, and schema."""

from .base_chunker import BaseChunker
from .config import ChunkConfig
from .recursive_chunker import RecursiveChunker
from .schema import Chunk, ChunkMetadata

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkConfig",
    "ChunkMetadata",
    "RecursiveChunker",
]
