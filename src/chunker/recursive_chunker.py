"""Recursive character text splitter — pure Python, no LangChain."""

from __future__ import annotations

import logging

from chunker.base_chunker import BaseChunker
from chunker.config import ChunkConfig
from chunker.schema import Chunk, ChunkMetadata
from loader.base_loader import LoadedDocument

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """Split text recursively using a hierarchy of separators.

    Tries each separator in order. When a separator produces pieces that
    are still too large, the next separator is tried on those pieces.
    This mirrors LangChain's ``RecursiveCharacterTextSplitter`` but uses
    only project-owned domain types.

    Args:
        config: Chunking configuration (size, overlap, separators).
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self._config = config or ChunkConfig()

    # ── Public API ───────────────────────────────────────────────────

    def chunk(self, document: LoadedDocument) -> list[Chunk]:
        """Split a loaded document into overlapping chunks.

        Args:
            document: A LoadedDocument produced by a loader.

        Returns:
            An ordered list of Chunk objects with metadata preserved.
        """
        raw_chunks = self._split_text(
            document.page_content, self._config.separators
        )

        chunks: list[Chunk] = []
        offset = 0
        for idx, text in enumerate(raw_chunks):
            start = document.page_content.find(text, offset)
            if start == -1:
                start = offset
            end = start + len(text)
            offset = start + 1

            chunks.append(
                Chunk(
                    content=text,
                    metadata=ChunkMetadata(
                        source=document.metadata.source,
                        page=document.metadata.page,
                        chunk_index=idx,
                    ),
                    start_index=start,
                    end_index=end,
                )
            )

        logger.debug(
            "Chunked document (%d chars) into %d chunks",
            len(document.page_content),
            len(chunks),
        )
        return chunks

    # ── Internals ────────────────────────────────────────────────────

    def _length(self, text: str) -> int:
        """Measure text length according to the configured function."""
        if self._config.length_function == "token":
            # Simple whitespace tokenisation; swap for a real tokeniser
            # (e.g. tiktoken) when token-level precision is needed.
            return len(text.split())
        return len(text)

    def _split_text(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """Recursively split *text* using the separator hierarchy."""
        final_chunks: list[str] = []

        # Find the best (first matching) separator
        separator = ""
        remaining_separators = separators
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_separators = []
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1 :]
                break

        # Split on the chosen separator
        splits = text.split(separator) if separator else list(text)

        # Merge small pieces, recurse on large ones
        good_splits: list[str] = []
        for piece in splits:
            if self._length(piece) < self._config.chunk_size:
                good_splits.append(piece)
            else:
                # Flush accumulated good splits first
                if good_splits:
                    final_chunks.extend(
                        self._merge_splits(good_splits, separator)
                    )
                    good_splits = []

                # Recurse if there are finer separators available
                if remaining_separators:
                    final_chunks.extend(
                        self._split_text(piece, remaining_separators)
                    )
                else:
                    final_chunks.append(piece)

        # Flush remaining
        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(
        self, splits: list[str], separator: str
    ) -> list[str]:
        """Merge small splits into chunks respecting size and overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        sep_len = self._length(separator)

        for piece in splits:
            piece_len = self._length(piece)
            total = current_len + piece_len + (sep_len if current else 0)

            if total > self._config.chunk_size and current:
                # Emit the current chunk
                chunk_text = separator.join(current)
                if chunk_text.strip():
                    chunks.append(chunk_text)

                # Keep overlap: drop from the front until within budget
                while (
                    current
                    and current_len > self._config.chunk_overlap
                ):
                    dropped = self._length(current[0]) + (
                        sep_len if len(current) > 1 else 0
                    )
                    current_len -= dropped
                    current.pop(0)

            current.append(piece)
            current_len += piece_len + (sep_len if len(current) > 1 else 0)

        # Final chunk
        if current:
            chunk_text = separator.join(current)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks
