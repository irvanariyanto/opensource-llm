# Open Source LLM

Run local LLM experiments with [Ollama](https://ollama.com) and Python — from notebooks and prompt engineering to a PDF-based RAG chatbot powered by [LangChain](https://www.langchain.com/) and [Chroma](https://www.trychroma.com/).

## Project structure

```
├── .gitignore
├── README.md
├── requirements.txt
├── LICENSE
├── .env.example
├── notebooks/
│   ├── 01_first_ollama.ipynb
│   └── 02_prompt_engineering.ipynb
├── src/
│   ├── rag_chatbot.py    # RAG pipeline (load PDF → embed → query)
│   └── rag_app.py        # Interactive CLI chat
├── data/
│   └── iso27001.pdf      # Place your PDF here (not tracked in git)
├── chroma_db/            # Generated vector store (not tracked in git)
├── docs/
│   └── notes.md
└── tests/
    └── test_basic.py
```

## Requirements

- [Ollama](https://ollama.com/download)
- Python 3.10+

## Setup

### 1. Install Ollama

Download and install from [ollama.com/download](https://ollama.com/download), then verify:

```bash
ollama --version
```

### 2. Pull models

This project uses a chat model and an embedding model:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### Useful Ollama commands

| Command | Description |
|---------|-------------|
| `ollama run <model>` | Download (if needed) and start a chat session |
| `ollama list` | List downloaded models |
| `ollama pull <model>` | Download a model without starting chat |
| `ollama rm <model>` | Remove a model |
| `/bye` | Exit chat session |

### Recommended chat models (light → heavy)

| Model | Size | Notes |
|-------|------|-------|
| `llama3.2:1b` | ~1.3 GB | Default for this repo |
| `gemma2:2b` | ~1.6 GB | Google |
| `qwen2.5:3b` | ~1.9 GB | Good multilingual support |
| `llama3.1:8b` | ~4.7 GB | More capable |
| `mistral:7b` | ~4.1 GB | Popular general-purpose model |

### 3. Python environment

```bash
python -m venv llm-env

# Linux / macOS
source llm-env/bin/activate

# Windows
llm-env\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

### 4. Add a PDF document

Place a PDF at `data/iso27001.pdf` (or change `PDF_PATH` in `src/rag_chatbot.py`). PDF files in `data/` are not committed to git.

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Chat model for answers |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for RAG |

## Run

Make sure Ollama is running, then:

```bash
# Notebooks
jupyter notebook notebooks/

# RAG chatbot (indexes PDF on first start, then interactive Q&A)
python -m src.rag_app

# Tests
pytest
```

On first launch, `rag_chatbot.py` loads the PDF, splits it into chunks, embeds them with Ollama, and stores the vectors in `chroma_db/`. Subsequent runs reuse the persisted store.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Connection refused | Start Ollama (`ollama serve` or open the Ollama app) |
| Model not found | `ollama pull llama3.2:1b` and `ollama pull nomic-embed-text` |
| PDF not found | Add your PDF to `data/iso27001.pdf` |
| `ModuleNotFoundError` | Activate the venv and run `pip install -r requirements.txt` |
| Slow first run | Initial indexing embeds every chunk — this is expected |
