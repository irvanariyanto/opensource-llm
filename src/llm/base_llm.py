"""Base abstraction for large language model interaction."""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base for all LLM providers.

    Implementations wrap a specific LLM backend (Ollama, OpenAI, etc.)
    behind a simple text-in / text-out contract.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response from a prompt.

        Args:
            prompt: The full prompt string (including any context).

        Returns:
            The generated text response.
        """
        ...
