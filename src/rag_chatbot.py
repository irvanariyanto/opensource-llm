import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA

# ===== Config =====
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
PDF_PATH = "data/iso27001.pdf"
PERSIST_DIR = "./chroma_db"

# ===== 1. Load PDF =====
print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# ===== 2. Split to chunks =====
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# ===== 3. Setup Embeddings (LangChain wrapper) =====
embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL)

# ===== 4. Embed and store in Chroma =====
print("Embedding and storing in Chroma...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)
print("Stored in Chroma")

# ===== 5. Setup Retriever =====
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ===== 6. Setup LLM =====
llm = ChatOllama(model=DEFAULT_MODEL, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# ===== 8. Interactive Chat Loop =====
def chat(question: str) -> dict:
    """Tanya ke RAG chain dan return jawaban + sumber."""
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [
            {
                "page": doc.metadata.get("page", "N/A"),
                "snippet": doc.page_content[:150] + "...",
            }
            for doc in result["source_documents"]
        ],
    }


