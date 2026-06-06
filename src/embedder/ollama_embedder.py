"""Ollama embedding adapter."""

import logging

from langchain_ollama import OllamaEmbeddings

from embedder.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class OllamaEmbedder(BaseEmbedder):
    """Embedding adapter backed by Ollama via langchain_ollama.

    Args:
        model: The Ollama embedding model name (e.g. ``"nomic-embed-text"``).
    """

    def __init__(self, model: str = "nomic-embed-text") -> None:
        self._model = model
        self._engine = OllamaEmbeddings(model=model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Ollama.

        Args:
            texts: Text strings to embed.

        Returns:
            A list of embedding vectors.
        """
        logger.debug("Embedding %d texts with model '%s'", len(texts), self._model)
        return self._engine.embed_documents(texts)
