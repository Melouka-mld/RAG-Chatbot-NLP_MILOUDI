"""
ingestion.py
============
Step 1 of the RAG pipeline.

What this file does:
  1. Loads all documents from the data/ folder
  2. Splits them into overlapping chunks
  3. Converts each chunk into a vector (embedding)
  4. Saves all vectors to a local FAISS database

Run this ONCE before starting the chatbot:
    python ingestion.py

After adding new documents to data/, run again to update:
    python ingestion.py --reset
"""

import argparse
import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ── Configuration ─────────────────────────────────────────────
DATA_PATH  = "data"        # folder with your source documents
FAISS_PATH = "faiss_db"    # where the vector database is saved
CHUNK_SIZE = 800           # characters per chunk
CHUNK_OVERLAP = 80         # overlap between consecutive chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model (FREE, runs locally)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents() -> list[Document]:
    """Load all .txt and .pdf files from the data/ folder."""
    documents = []

    # Load .txt files
    try:
        txt_loader = DirectoryLoader(
            DATA_PATH, glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        documents += txt_loader.load()
    except Exception as e:
        print(f"  ⚠️  TXT loading warning: {e}")

    # Load .pdf files
    try:
        pdf_loader = DirectoryLoader(
            DATA_PATH, glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents += pdf_loader.load()
    except Exception as e:
        print(f"  ⚠️  PDF loading warning: {e}")

    print(f"  ✅ Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into small overlapping chunks.

    Why chunking?
    - LLMs have a context window limit
    - We only pass the TOP-3 most relevant chunks per question
    - Smaller chunks = more precise retrieval
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  ✅ Split into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_faiss(chunks: list[Document], reset: bool = False):
    """
    Build or update the FAISS vector database.

    Each chunk is converted to a 384-dim vector by the embedding model.
    FAISS stores all vectors for fast similarity search.
    """
    if reset and os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)
        print(f"  🗑️  Deleted old database at '{FAISS_PATH}/'")

    embedding_model = get_embedding_model()

    if os.path.exists(FAISS_PATH):
        print(f"  📂 Updating existing database at '{FAISS_PATH}/'...")
        db = FAISS.load_local(FAISS_PATH, embedding_model,
                              allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        print(f"  🏗️  Building new database at '{FAISS_PATH}/'...")
        db = FAISS.from_documents(chunks, embedding_model)

    db.save_local(FAISS_PATH)
    print(f"  ✅ Database saved! Total vectors: {db.index.ntotal}")
    return db


def main(reset: bool = False):
    print("\n" + "="*55)
    print("  RAG INGESTION PIPELINE — MILOUDI NLP 2025")
    print("="*55)

    print("\n📂 Step 1: Loading documents from data/...")
    documents = load_documents()

    print("\n✂️  Step 2: Splitting into chunks...")
    chunks = split_documents(documents)

    print("\n🧮 Step 3: Embedding & storing in FAISS...")
    print("  ⏳ First run downloads ~80MB model — please wait...")
    db = build_faiss(chunks, reset=reset)

    print("\n" + "="*55)
    print("  ✅ INGESTION COMPLETE!")
    print(f"  📊 {db.index.ntotal} vectors stored in '{FAISS_PATH}/'")
    print("  👉 Now run: streamlit run chatbot_rag.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Delete and rebuild the database from scratch")
    args = parser.parse_args()
    main(reset=args.reset)
