# app.py
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# LLM (answers): Groq
from langchain_groq import ChatGroq

# Embeddings: OpenAI ONLY
from langchain_openai import OpenAIEmbeddings

# RAG plumbing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)

# ------------------ Setup ------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Optional LangSmith
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Document Q&A"

UPLOAD_DIR = Path("Uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG - Document Q&A", page_icon="📄", layout="wide")
st.title("📄 RAG — Chat with your document")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.subheader("Configuration")
    # Ask user for OpenAI API key (masked)
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Used for embeddings only.")
    top_k = st.slider("Top-K chunks", 2, 10, 4)
    chunk_size = st.slider("Chunk size (chars)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, step=20)
    if st.button("Clear session"):
        for k in ("vectors", "docs", "messages"):
            st.session_state.pop(k, None)
        st.rerun()

# ------------------ Upload ------------------
uploaded_files = st.file_uploader(
    "Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

def save_uploads(files):
    saved = []
    for f in files:
        dest = UPLOAD_DIR / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        saved.append(dest)
    return saved

def load_documents(paths):
    docs = []
    for p in paths:
        try:
            if p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
            elif p.suffix.lower() == ".docx":
                docs.extend(Docx2txtLoader(str(p)).load())
            elif p.suffix.lower() == ".txt":
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
        except Exception as e:
            st.warning(f"Skipping {p.name}: {e}")
    return docs

def build_index(docs, *, chunk_size: int, chunk_overlap: int, openai_api_key: str):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    # Make sure downstream libs can see the key too
    os.environ["OPENAI_API_KEY"] = openai_api_key

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # OpenAI embeddings only
    embeddings = OpenAIEmbeddings(
        # choose your model; 3-small is cheap, 3-large is best quality
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )

    vs = FAISS.from_documents(chunks, embeddings)
    return vs, chunks

# ------------------ LLM (Groq) ------------------
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing in environment.")
    st.stop()

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
)

prompt_template = ChatPromptTemplate.from_template(
    """
You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say you don't find it in the document.
Cite sources like [source: {{file}} p.{{page}}] after the relevant sentence.

Context:
{context}

Question: {input}
"""
)

# ------------------ Session ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "docs" not in st.session_state:
    st.session_state.docs = []

# ------------------ Build index ------------------
if uploaded_files and st.button("Build / Rebuild Index"):
    saved_paths = save_uploads(uploaded_files)
    with st.spinner("Loading documents…"):
        docs = load_documents(saved_paths)
        if not docs:
            st.error("No readable documents found.")
            st.stop()
        st.session_state.docs = docs

    with st.spinner("Embedding (OpenAI) & indexing…"):
        st.session_state.vectors, _ = build_index(
            st.session_state.docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            openai_api_key=openai_api_key,
        )
    st.success(f"Indexed {len(st.session_state.docs)} document(s).")

# ------------------ Chat UI ------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask something about your document…")

if user_q:
    if st.session_state.vectors is None:
        st.warning("Please upload files and click **Build / Rebuild Index** first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_q})

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": top_k})
        rag_chain = create_retrieval_chain(retriever, document_chain)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                t0 = time.perf_counter()
                resp = rag_chain.invoke({"input": user_q})
                dt = time.perf_counter() - t0

                st.markdown(resp.get("answer", "_No answer_"))
                st.caption(f"Response time: {dt:.2f}s")

                with st.expander("Show retrieved sources"):
                    for i, doc in enumerate(resp.get("context", []), start=1):
                        meta = doc.metadata or {}
                        file = meta.get("source", "unknown")
                        page = meta.get("page", meta.get("page_number", None))
                        label = f"**{i}. {Path(file).name}**"
                        if page is not None:
                            # PyPDFLoader gives 0-based page numbers
                            label += f" — page {int(page)+1 if isinstance(page, int) else page}"
                        st.markdown(label)
                        st.write(doc.page_content[:800] + ("…" if len(doc.page_content) > 800 else ""))

        st.session_state.messages.append({"role": "assistant", "content": resp.get("answer", "")})