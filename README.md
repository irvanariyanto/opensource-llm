# Open Source LLM

Run local LLM experiments with [Ollama](https://ollama.com) and Python.

## Requirements

- [Ollama](https://ollama.com/download)
- Python 3.10+

## Setup

### 1. Install Ollama

Download and install from [ollama.com/download](https://ollama.com/download), then verify:

```bash
ollama --version
```

### 2. Pull the model

```bash
ollama pull llama3.2:1b
```

### Useful Ollama commands

| Command | Description |
|---------|-------------|
| `ollama run <model>` | Download (if needed) and start a chat session |
| `ollama list` | List downloaded models |
| `ollama pull <model>` | Download a model without starting chat |
| `ollama rm <model>` | Remove a model |
| `/bye` | Exit chat session |

### Recommended models (light → heavy)

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

pip install transformers torch ollama langchain langchain-community
```

## Run

Make sure Ollama is running, then:

```bash
python first_llm.py
python promtp_engineering.py
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Connection refused | Start Ollama (`ollama serve` or open the Ollama app) |
| Model not found | `ollama pull llama3.2:1b` |
| `ModuleNotFoundError: ollama` | Activate the venv and run `pip install ollama` |
