"""Ollama LLM adapter."""

import logging

from langchain_ollama import ChatOllama

from llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """LLM adapter backed by Ollama via langchain_ollama.

    Args:
        model: The Ollama model name (e.g. ``"llama3.2:1b"``).
        temperature: Sampling temperature (0 = deterministic).
    """

    def __init__(
        self,
        model: str = "llama3.2:1b",
        temperature: float = 0,
    ) -> None:
        self._model = model
        self._engine = ChatOllama(model=model, temperature=temperature)

    def generate(self, prompt: str) -> str:
        """Generate a response using Ollama.

        Args:
            prompt: The full prompt string.

        Returns:
            The generated text response.
        """
        logger.debug("Generating response with model '%s'", self._model)
        response = self._engine.invoke(prompt)
        return str(response.content)
