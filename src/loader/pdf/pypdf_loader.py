"""PDF document loader using PyPDF."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader as _LCPyPDF

from loader.base_loader import BaseLoader, DocumentMetadata, LoadedDocument

logger = logging.getLogger(__name__)


class PyPDFLoader(BaseLoader):
    """Loads PDF documents via the PyPDF library."""

    def load(self, source: str) -> list[LoadedDocument]:
        """Load all pages from a PDF file.

        Args:
            source: File path to the PDF document.

        Returns:
            A list of LoadedDocument, one per page.

        Raises:
            FileNotFoundError: If the source path does not exist.
            ValueError: If the file is not a PDF.
            RuntimeError: If PDF parsing fails.
        """
        path = Path(source)

        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {source}")
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {source}")

        try:
            pages = _LCPyPDF(str(path)).load()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse PDF '{source}'") from exc

        logger.info("Loaded %d pages from %s", len(pages), source)
        return [
            LoadedDocument(
                page_content=page.page_content,
                metadata=DocumentMetadata(
                    source=str(path),
                    page=page.metadata.get("page"),
                    extra={
                        k: v
                        for k, v in page.metadata.items()
                        if k not in ("source", "page")
                    },
                ),
            )
            for page in pages
        ]

    @staticmethod
    def supports(source: str) -> bool:
        """Check if the source is a PDF file."""
        return source.lower().endswith(".pdf")
