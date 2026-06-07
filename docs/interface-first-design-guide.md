# How to Think Like an Architect — Interface-First Design Guide

A step-by-step thinking process you can apply to **any** project, using the RAG pipeline as the running example.

---

## The 5-Phase Thinking Process

```
1. Name the Responsibilities
        ↓
2. Draw the Boundaries
        ↓
3. Define the Contracts (Interfaces)
        ↓
4. Implement the Adapters
        ↓
5. Wire it Together
```

---

## Phase 1 — Name the Responsibilities

> **Question to ask:** _"If I explain what my code does to a non-programmer, what VERBS do I use?"_

### How I thought about the RAG project

I read `rag_chatbot.py` and underlined every **distinct action**:

| Line  | What the code does              | Verb / Responsibility  |
| ----- | ------------------------------- | ---------------------- |
| 20-22 | Open a PDF and read pages       | **Loading**            |
| 26-32 | Split text into smaller pieces  | **Chunking**           |
| 35    | Convert text to numbers         | **Embedding**          |
| 39-43 | Save and search those numbers   | **Storing / Retrieving** |
| 50    | Ask an AI to answer             | **Generating**         |

Five verbs → five responsibilities. That's it. This is the most important step.

### How to practice this

1. Open any file in your project
2. For every block of code, write a **one-verb summary** in the margin
3. If two blocks share the same verb → they belong together
4. If one block has two verbs → it should be split

> **💡 The "explain to a friend" test:** If you say _"this function loads a PDF AND splits it into chunks AND embeds them"_, every AND is a boundary you're crossing. Each AND should be a separate module.

---

## Phase 2 — Draw the Boundaries

> **Question to ask:** _"Which parts can I replace independently?"_

### The Dependency Direction Rule

```
Things that CHANGE often          Things that are STABLE
(Which PDF library?          →    ("I need to load docs"
 Which embedding API?              "I need to embed text"
 Which vector DB?)                  "I need to search vectors")

Unstable depends on Stable.
Stable NEVER depends on Unstable.
```

**Stable things (interfaces) should NEVER import unstable things (libraries).**

### How I drew boundaries in the RAG project

I asked myself: _"What if the user wants to..."_

| "What if I want to..."              | What changes                    | What stays the same                       |
| ------------------------------------ | ------------------------------- | ----------------------------------------- |
| Use a different PDF library          | `PyPDFLoader`                   | The concept of "load a document"          |
| Switch from Ollama to OpenAI         | `OllamaEmbedder`, `OllamaLLM`  | The concept of "embed text", "generate answer" |
| Use Pinecone instead of Chroma       | `ChromaVectorStore`             | The concept of "store and search chunks"  |
| Change chunk size                    | `ChunkConfig`                   | The chunking algorithm                    |

The **left column** = adapters (concrete, swappable)
The **right column** = interfaces (abstract, stable)

### The layer cake

```
┌─────────────────────────────────────────┐
│         Entry Points (consumers)        │  ← rag_app.py, streamlit
│         Know HOW to wire things         │
├─────────────────────────────────────────┤
│         Pipeline (orchestrator)         │  ← RAGPipeline
│         Knows the WORKFLOW              │
│         Imports only INTERFACES         │
├─────────────────────────────────────────┤
│         Interfaces (ports/contracts)    │  ← BaseLoader, BaseLLM, etc.
│         Know WHAT, never HOW            │
├─────────────────────────────────────────┤
│         Domain Types (data)             │  ← Chunk, LoadedDocument, etc.
│         Pure data, zero logic           │
└─────────────────────────────────────────┘

Rule: Each layer can ONLY import from layers BELOW it.
      Never sideways. Never upward.
```

> **⚠️ The one rule that prevents spaghetti:** Arrows only point downward. If you ever feel the need to import upward, you're missing an interface.

---

## Phase 3 — Define the Contracts (Interfaces)

> **Question to ask:** _"What is the MINIMUM a caller needs to know?"_

### The recipe for writing a good interface

```python
# 1. Start with the verb from Phase 1
#    Verb: "embed"

# 2. What goes IN? (simplest possible type)
#    Input: list of strings

# 3. What comes OUT? (simplest possible type)
#    Output: list of float vectors

# 4. Write ONLY that — nothing else
class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...
```

### Three rules to always follow

**Rule 1: Use your own types, not library types**

```python
# ❌ BAD — caller must know about LangChain
def chunk(self, document: Document) -> list[Document]:  # LangChain types leak out

# ✅ GOOD — caller only knows YOUR types
def chunk(self, document: LoadedDocument) -> list[Chunk]:  # your own types
```

**Rule 2: Interfaces should be boring**

An interface should fit in your head. If it has more than 3-4 methods, it's doing too much — split it.

```python
# ❌ Too much — this is a God Interface
class BaseVectorStore(ABC):
    def add_chunks(self, ...): ...
    def search(self, ...): ...
    def delete(self, ...): ...
    def update(self, ...): ...
    def backup(self, ...): ...
    def get_stats(self, ...): ...

# ✅ Just right — only the core contract
class BaseVectorStore(ABC):
    def add_chunks(self, chunks, embeddings): ...
    def search(self, query_embedding, k): ...
```

**Rule 3: Return domain types, not dicts**

```python
# ❌ Fragile — caller has to guess the keys
def query(self, question: str) -> dict: ...

# ✅ Self-documenting — IDE shows you exactly what you get
def query(self, question: str) -> QueryResult: ...

@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[SourceReference]
```

---

## Phase 4 — Implement the Adapters

> **Question to ask:** _"How do I make THIS library fit THAT interface?"_

This is the only place where you import third-party libraries. The pattern is always the same:

```python
# adapter = interface + library

from your_interface import BaseEmbedder          # YOUR contract
from langchain_ollama import OllamaEmbeddings    # THEIR library

class OllamaEmbedder(BaseEmbedder):              # Adapter bridges the two
    def __init__(self, model: str):
        self._engine = OllamaEmbeddings(model=model)  # library detail hidden

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._engine.embed_documents(texts)     # translate call
```

### The adapter checklist

- [ ] Does it import the interface? ✅
- [ ] Does it import the library? ✅
- [ ] Does the `__init__` accept only simple config (strings, ints)? ✅
- [ ] Do the public methods match the interface exactly? ✅
- [ ] Is the library object private (`self._engine`, not `self.engine`)? ✅

---

## Phase 5 — Wire it Together (Composition Root)

> **Question to ask:** _"Where do I choose WHICH adapter to use?"_

The answer: **one place, and one place only.** This is called the **Composition Root**.

```python
# rag_chatbot.py — THE ONLY FILE that knows all concrete classes

def build_pipeline() -> RAGPipeline:
    return RAGPipeline(
        loader=PyPDFLoader(),                    # ← choice: PyPDF
        chunker=RecursiveChunker(ChunkConfig()),  # ← choice: recursive
        embedder=OllamaEmbedder(model="..."),    # ← choice: Ollama
        vector_store=ChromaVectorStore(),         # ← choice: Chroma
        llm=OllamaLLM(model="..."),              # ← choice: Ollama
    )
```

> **💡 How to know you did it right:** The composition root is the only file where you see all the concrete class names. Every other file works with interfaces only.

---

## The Decision Flowchart

Use this when you're building any new feature:

```
          ┌──────────────────────────────────┐
          │ I need to add new functionality  │
          └──────────────┬───────────────────┘
                         │
                ┌────────▼────────┐
                │ Does it talk to │
                │ an external     │──── Yes ──→ Create an INTERFACE
                │ system?         │              in the ports layer
                └────────┬────────┘                     │
                         │ No                           ▼
                ┌────────▼────────┐           Create an ADAPTER
                │ Is it pure      │           that implements it
                │ business logic? │                     │
                └────────┬────────┘                     ▼
                    │         │              Wire it in the
                   Yes        No             COMPOSITION ROOT
                    │         │
                    ▼         ▼
              Domain /    Utility
              Pipeline    module
              layer
```

---

## Common Mistakes (and How to Avoid Them)

### Mistake 1: "I'll just use the library directly"

```python
# ❌ This is how it starts — "just one import"
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectors = embeddings.embed_documents(texts)
```

Then you do this in 5 files. Then LangChain releases a breaking change. Now you're editing 5 files.

**Fix:** One interface, one adapter, one line to change.

### Mistake 2: "My interface is just a wrapper"

If your interface method signatures look exactly like the library — that's fine! The value isn't in being different, it's in being **stable**.

```python
# This looks "useless" but it's INCREDIBLY valuable:
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...
```

Why? Because when you switch from Ollama to OpenAI next month, you change ONE file instead of twenty.

### Mistake 3: "I'll add the interface later"

No. **Interface first, implementation second.** Writing the interface forces you to think about what the caller actually needs — before you get lost in library details.

### Mistake 4: Putting everything in one file

Before:
```
src/
  rag_chatbot.py  ← 200 lines doing EVERYTHING
```

After:
```
src/
  loader/          ← 1 responsibility
  chunker/         ← 1 responsibility
  embedder/        ← 1 responsibility
  vector_store/    ← 1 responsibility
  llm/             ← 1 responsibility
  pipeline/        ← orchestration only
  rag_chatbot.py   ← wiring only
```

> **💡 Each folder answers ONE question:** "How do we load?", "How do we chunk?", "How do we embed?" — never two questions in one folder.

---

## Quick Checklist for Any New Project

Before you write code, go through these steps:

1. **□ List the verbs** — what does my system DO?
2. **□ One verb = one module** — create a folder for each
3. **□ Write the interface first** — `BaseXxx` with `@abstractmethod`
4. **□ Define your data types** — `@dataclass` for inputs/outputs
5. **□ Implement the adapter** — wrap the library behind your interface
6. **□ Build the orchestrator** — imports ONLY interfaces
7. **□ Wire in the composition root** — the ONLY place that picks concrete classes

If you follow this every time, you'll produce code that:

- **Reads like documentation** (the interface tells you what the system does)
- **Survives library changes** (swap one adapter, done)
- **Is testable** (mock any interface in unit tests)
- **Scales** (new team members understand the architecture in minutes)

---

## Applying This to a Real Example

Here is the final file structure from this project, annotated:

```
src/
├── chunker/                         # Responsibility: CHUNKING
│   ├── base_chunker.py              #   ← Interface (port)
│   ├── config.py                    #   ← Domain type
│   ├── recursive_chunker.py         #   ← Adapter (pure Python)
│   └── schema.py                    #   ← Domain types
│
├── embedder/                        # Responsibility: EMBEDDING
│   ├── base_embedder.py             #   ← Interface (port)
│   └── ollama_embedder.py           #   ← Adapter (LangChain)
│
├── llm/                             # Responsibility: GENERATING
│   ├── base_llm.py                  #   ← Interface (port)
│   └── ollama_llm.py                #   ← Adapter (LangChain)
│
├── loader/                          # Responsibility: LOADING
│   ├── base_loader.py               #   ← Interface (port) + domain types
│   └── pdf/
│       └── pypdf_loader.py          #   ← Adapter (PyPDF)
│
├── vector_store/                    # Responsibility: STORING/RETRIEVING
│   ├── base_vector_store.py         #   ← Interface (port) + SearchResult
│   └── chroma_store.py              #   ← Adapter (ChromaDB)
│
├── pipeline/                        # ORCHESTRATION (imports only ports)
│   ├── rag_pipeline.py              #   ← Workflow: ingest + query
│   └── schema.py                    #   ← QueryResult, SourceReference
│
├── rag_chatbot.py                   # COMPOSITION ROOT (wires adapters)
├── rag_app.py                       # Entry point: CLI
└── rag_streamlite.py                # Entry point: Web UI
```

Notice the pattern: every folder has **one interface** and **one or more adapters**. The pipeline folder has **zero library imports**. The composition root is the **only file** that knows which concrete classes to use.

---

## Recommended Reading

| Resource                                               | Why                                         |
| ------------------------------------------------------ | ------------------------------------------- |
| _Clean Architecture_ — Robert C. Martin                | The full theory behind what we did           |
| _Architecture Patterns with Python_ — Percival & Gregory | Ports & Adapters in Python specifically     |
| _Cosmic Python_ (free online: cosmicpython.com)        | Same authors, free web version               |
| SOLID principles (any tutorial)                        | The five rules underlying all of this        |
