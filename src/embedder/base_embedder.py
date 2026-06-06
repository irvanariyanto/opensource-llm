"""Base abstraction for text embedding."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base for all embedding providers.

    Implementations convert text strings into dense vector representations
    suitable for similarity search.
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert a batch of text strings to embedding vectors.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text.
            Each vector is a list of floats.
        """
        ...

    def embed_text(self, text: str) -> list[float]:
        """Convenience method: embed a single text string.

        Args:
            text: The text string to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        return self.embed_texts([text])[0]
