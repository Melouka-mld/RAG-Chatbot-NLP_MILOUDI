"""
retrieval.py
============
Step 2 of the RAG pipeline — the "R" in RAG.

What this file does:
  - Loads the FAISS database from disk
  - Converts a user question into a vector
  - Finds the TOP-K most similar chunks
  - Returns those chunks as context for the LLM

This is NOT keyword search — it finds chunks with
similar MEANING even if the words are different.

Example:
  Question: "bad experience"
  Finds: "disappointed with product quality" ✓
  (same meaning, different words)

Usage:
  from retrieval import retrieve_docs
  docs = retrieve_docs("What do customers say about delivery?")
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ── Configuration ─────────────────────────────────────────────
FAISS_PATH      = "faiss_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K           = 3   # number of chunks to retrieve per query


def load_db() -> FAISS:
    """Load the FAISS vector store from disk."""
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(
            f"❌ Database not found at '{FAISS_PATH}/'\n"
            "   Run: python ingestion.py"
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(FAISS_PATH, embeddings,
                            allow_dangerous_deserialization=True)


def retrieve_docs(query: str, k: int = TOP_K) -> list[Document]:
    """
    Find the k most relevant chunks for a given query.

    Args:
        query : the user's question
        k     : number of chunks to return (default=3)

    Returns:
        list of Document objects with .page_content and .metadata
    """
    db   = load_db()
    docs = db.similarity_search(query, k=k)
    return docs


def retrieve_with_scores(query: str, k: int = TOP_K):
    """
    Find chunks with their similarity scores.

    Returns: list of (Document, score) tuples
    Lower score = MORE similar (L2 distance)
    """
    db      = load_db()
    results = db.similarity_search_with_score(query, k=k)
    return results


def format_context(docs: list[Document]) -> str:
    """Merge retrieved chunks into one context string for the LLM."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── Quick test when run directly ───────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
            "What do customers say about delivery?"

    print(f"\n🔍 Query: {query}")
    print("="*55)

    results = retrieve_with_scores(query, k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[Chunk {i}] Similarity score: {score:.4f}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
