"""Domain types for RAG pipeline query results."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceReference:
    """A reference to the source material that informed an answer.

    Attributes:
        page: The page number in the original document (if available).
        snippet: A short text excerpt from the source chunk.
        score: Similarity score from the vector search.
    """

    page: int | None
    snippet: str
    score: float = 0.0


@dataclass(frozen=True)
class QueryResult:
    """The result of a RAG pipeline query.

    Attributes:
        answer: The generated answer text.
        sources: References to the source chunks used.
    """

    answer: str
    sources: list[SourceReference] = field(default_factory=list)
