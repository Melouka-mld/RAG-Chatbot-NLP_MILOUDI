"""
chatbot_rag.py
==============
Step 3 of the RAG pipeline — the Streamlit chatbot UI.

This is the main application file. It:
  1. Loads the FAISS database and the LLM
  2. For each user message: retrieves top-3 chunks → builds prompt → calls LLM
  3. Displays the answer with source references

Run:
    streamlit run chatbot_rag.py

Requirements:
  - Run ingestion.py first to build the database
  - Set HUGGINGFACEHUB_API_TOKEN in .env or environment
"""

import os
import streamlit as st
from dotenv import load_dotenv

from retrieval import retrieve_docs, format_context
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# ── Load environment variables ─────────────────────────────────
load_dotenv()

# ── Page configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot — MILOUDI NLP 2025",
    page_icon="🤖",
    layout="wide",
)

# ── Prompt template ────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about customer feedback.
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say: "I don't have enough information to answer that."
Do NOT make up information that is not in the context.

Context:
{context}

Question: {question}

Answer:"""


# ── Load LLM (cached so it loads only once) ────────────────────
@st.cache_resource
def load_llm():
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        st.error("❌ HuggingFace token not found! Set HUGGINGFACEHUB_API_TOKEN in .env")
        st.stop()
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 256},
        huggingfacehub_api_token=token,
    )


def get_answer(question: str) -> tuple[str, list]:
    """
    Full RAG pipeline:
      1. Retrieve top-3 relevant chunks from FAISS
      2. Format them as context
      3. Build the prompt
      4. Call the LLM
      5. Return answer + source chunks
    """
    # Step 1 — Retrieve
    docs    = retrieve_docs(question, k=3)
    context = format_context(docs)

    # Step 2 — Build prompt
    prompt_template = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    prompt = prompt_template.format(context=context, question=question)

    # Step 3 — Generate answer
    llm    = load_llm()
    answer = llm.invoke(prompt)

    return answer, docs


# ── UI Layout ──────────────────────────────────────────────────
def main():

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg",
                 width=50)
        st.title("ℹ️ Model Info")
        st.markdown("""
        | Component | Details |
        |-----------|---------|
        | **LLM** | google/flan-t5-large |
        | **Params** | ~780M |
        | **Embeddings** | all-MiniLM-L6-v2 |
        | **Vector DB** | FAISS (local) |
        | **Top-K** | 3 chunks |
        | **Chunk size** | 800 chars |
        """)
        st.divider()
        st.markdown("**💡 Example questions:**")
        examples = [
            "What do customers say about delivery?",
            "What are the main complaints?",
            "How satisfied are customers with quality?",
            "How was the customer service described?",
            "How long did the refund take?",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.example_question = ex

        st.divider()
        st.markdown("**📊 Dataset**")
        st.caption("data/comments.txt — Customer e-commerce reviews")

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Main panel ─────────────────────────────────────────────
    st.title("🤖 RAG Chatbot")
    st.caption("NLP Module Project 2025 — MILOUDI — Melouka-mld/RAG-Chatbot-NLP_MILOUDI")
    st.markdown("""
    Ask questions about the **customer comments dataset**.
    The chatbot retrieves relevant pieces of the document and answers based **only** on what is written there.
    """)
    st.divider()

    # ── Chat history ───────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Handle example button click ────────────────────────────
    if "example_question" in st.session_state:
        question = st.session_state.pop("example_question")
    else:
        question = None

    # ── Chat input ─────────────────────────────────────────────
    user_input = st.chat_input("Ask a question about the customer comments...")
    if user_input:
        question = user_input

    # ── Process question ───────────────────────────────────────
    if question:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate and show answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating answer..."):
                try:
                    answer, source_docs = get_answer(question)

                    st.markdown(answer)

                    # Show sources in expander
                    with st.expander("📚 Sources retrieved (click to expand)"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.caption(doc.page_content[:300] + "...")
                            st.divider()

                    full_response = f"{answer}\n\n*(Based on {len(source_docs)} retrieved chunks)*"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )

                except FileNotFoundError as e:
                    st.error(str(e))
                    st.info("👉 Run `python ingestion.py` first to build the database!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
