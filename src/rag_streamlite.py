import os

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv  

load_dotenv()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

st.title("📚 RAG Chatbot")

@st.cache_resource
def load_chain():
    embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=DEFAULT_MODEL, temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
chain = load_chain()

question = st.text_input("Tanyakan sesuatu tentang dokumen:")
if question:
    with st.spinner("Mencari jawaban..."):
        result = chain.invoke({"query": question})
        st.write(result["result"])
        st.write("-" * 60)
        st.write("Sumber:")
        for doc in result["source_documents"]:
            st.write(f"Halaman {doc.metadata.get('page', 'N/A')}:")
            st.write(doc.page_content[:150] + "...")