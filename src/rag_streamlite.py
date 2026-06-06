"""Streamlit web interface for the RAG chatbot."""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Late imports to avoid import-time side effects from adapters
from chunker.config import ChunkConfig  # noqa: E402
from chunker.recursive_chunker import RecursiveChunker  # noqa: E402
from embedder.ollama_embedder import OllamaEmbedder  # noqa: E402
from llm.ollama_llm import OllamaLLM  # noqa: E402
from loader.pdf.pypdf_loader import PyPDFLoader  # noqa: E402
from pipeline.rag_pipeline import RAGPipeline  # noqa: E402
from vector_store.chroma_store import ChromaVectorStore  # noqa: E402

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

st.title("📚 RAG Chatbot")


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    """Build and cache the RAG pipeline (runs once per Streamlit session)."""
    pipeline = RAGPipeline(
        loader=PyPDFLoader(),
        chunker=RecursiveChunker(
            ChunkConfig(chunk_size=500, chunk_overlap=50),
        ),
        embedder=OllamaEmbedder(model=DEFAULT_EMBEDDING_MODEL),
        vector_store=ChromaVectorStore(persist_dir="./chroma_db"),
        llm=OllamaLLM(model=DEFAULT_MODEL),
    )
    pipeline.ingest(os.getenv("PDF_PATH", "data/iso27001.pdf"))
    return pipeline


pipeline = load_pipeline()

question = st.text_input("Ask a question about the document:")
if question:
    with st.spinner("Searching for an answer..."):
        result = pipeline.query(question)
        st.write(result.answer)
        st.write("-" * 60)
        st.write("Sources:")
        for src in result.sources:
            page = src.page if src.page is not None else "N/A"
            st.write(f"Page {page} (score: {src.score:.3f}):")
            st.write(src.snippet)