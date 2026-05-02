# 🤖 RAG Chatbot — NLP Module 2025

> Question-Answering chatbot using **Retrieval-Augmented Generation (RAG)**  
> Built with LangChain · FAISS · HuggingFace · Gradio

**Student:** MILOUDI · **GitHub:** [Melouka-mld](https://github.com/Melouka-mld)

---

##  What is RAG?

```
 DOCUMENTS (data/)
        ↓
   Cut into chunks (800 chars)
        ↓
   Convert to vectors (embeddings) → stored in FAISS
        ↓
   User asks a question
        ↓
   Find TOP-3 most relevant chunks
        ↓
   LLM reads chunks → generates answer
```

---

##  Project Structure

```
RAG-Chatbot-NLP_MILOUDI/
├── data/
│   └── comments.txt          ← Personal dataset (customer comments)
├── ingestion.py              ← Step 1: Load → Chunk → Embed → Store
├── retrieval.py              ← Step 2: Question → FAISS → Top-3 chunks
├── chatbot_rag.py            ← Step 3: Gradio UI chatbot
├── MILOUDI_RAG_CHATBOT_2025.ipynb  ← Main Colab notebook
├── evaluation_results.csv    ← Evaluation on personal dataset
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

##  Tech Stack

| Component | Tool |
|-----------|------|
| Framework | LangChain |
| Vector DB | FAISS (local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | mistralai/Mistral-7B-Instruct-v0.3 (FREE) |
| UI | Gradio |

---

##  How to Run (Google Colab)

### 1. Open the notebook
Upload `MILOUDI_RAG_CHATBOT_2025.ipynb` to Google Colab

### 2. Run Cell 1 — install libraries
After it finishes: **Runtime → Restart session → OK**

### 3. Run Cell 2 — clone repo + set token
Get free token at: https://huggingface.co/settings/tokens

### 4. Run Cell 3 — build FAISS database
Loads `data/comments.txt` → chunks → embeddings → saves `faiss_db/`

### 5. Run Cell 7 — launch chatbot
A public gradio.live link appears → click it → chatbot is ready!

---

##  LLM Details

| Property | Value |
|----------|-------|
| Model | mistralai/Mistral-7B-Instruct-v0.3 |
| Architecture | Transformer decoder (Mistral) |
| Parameters | ~7 Billion |
| Training | Instruction fine-tuning |
| Cost | FREE via HuggingFace Inference API |
| Link | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 |

> **Note:** google/flan-t5 was removed from the free HuggingFace API in 2025.
> The project uses Mistral-7B-Instruct with automatic fallback to RAG retrieval mode.

---

##  Embedding Model

| Property | Value |
|----------|-------|
| Model | sentence-transformers/all-MiniLM-L6-v2 |
| Output | 384-dimensional normalized vectors |
| Runs | Locally in Colab — no API key needed |

---

##  Dataset

**File:** `data/comments.txt`
**Content:** E-commerce customer comments including:
- Product reviews (positive, neutral, negative)
- Shipping and delivery feedback
- Customer service experiences
- Rating statistics summary

---

##  Problems Encountered

| Problem | Solution |
|---------|----------|
| numpy.char conflict in Colab | Reinstall numpy==1.26.4, restart runtime |
| flan-t5 removed from free API (2025) | Migrated to Mistral-7B-Instruct-v0.3 |
| HTTP 403 on HuggingFace | New token with correct permissions |

---

##  Author

**MILOUDI** · NLP Module · 2025  
GitHub: [@Melouka-mld](https://github.com/Melouka-mld)  
Repository: https://github.com/Melouka-mld/RAG-Chatbot-NLP_MILOUDI
