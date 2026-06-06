"""Composition root — wires concrete adapters into the RAG pipeline."""

import logging
import os

from chunker.config import ChunkConfig
from chunker.recursive_chunker import RecursiveChunker
from embedder.ollama_embedder import OllamaEmbedder
from llm.ollama_llm import OllamaLLM
from loader.pdf.pypdf_loader import PyPDFLoader
from pipeline.rag_pipeline import RAGPipeline
from pipeline.schema import QueryResult
from vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
PDF_PATH = os.getenv("PDF_PATH", "data/iso27001.pdf")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


# ── Factory ──────────────────────────────────────────────────────────

def build_pipeline(
    pdf_path: str = PDF_PATH,
    persist_dir: str = PERSIST_DIR,
    model: str = DEFAULT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> RAGPipeline:
    """Build a fully wired RAG pipeline with concrete adapters.

    This is the **only place** where concrete implementations are chosen.
    The RAGPipeline itself depends only on abstract interfaces.

    Args:
        pdf_path: Path to the PDF document to ingest.
        persist_dir: Directory for the Chroma vector store.
        model: Ollama LLM model name.
        embedding_model: Ollama embedding model name.

    Returns:
        A configured RAGPipeline ready for ingest/query.
    """
    pipeline = RAGPipeline(
        loader=PyPDFLoader(),
        chunker=RecursiveChunker(
            ChunkConfig(chunk_size=500, chunk_overlap=50),
        ),
        embedder=OllamaEmbedder(model=embedding_model),
        vector_store=ChromaVectorStore(persist_dir=persist_dir),
        llm=OllamaLLM(model=model),
    )

    # Ingest the document
    num_chunks = pipeline.ingest(pdf_path)
    logger.info("Pipeline ready — %d chunks ingested", num_chunks)

    return pipeline


# ── Lazy singleton ───────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """Lazily initialise the pipeline on first use."""
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


# ── Public API ───────────────────────────────────────────────────────

def chat(question: str) -> dict:
    """Query the RAG pipeline and return the answer with sources.

    Maintains backwards compatibility with the original dict-based API.

    Args:
        question: The user's natural-language question.

    Returns:
        A dict with keys ``"answer"`` (str) and ``"sources"`` (list of dicts
        containing ``"page"``, ``"snippet"``, and ``"score"``).
    """
    result: QueryResult = _get_pipeline().query(question)

    return {
        "answer": result.answer,
        "sources": [
            {
                "page": src.page if src.page is not None else "N/A",
                "snippet": src.snippet,
                "score": round(src.score, 3),
            }
            for src in result.sources
        ],
    }
