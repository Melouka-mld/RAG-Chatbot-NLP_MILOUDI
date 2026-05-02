# 🤖 RAG Chatbot — NLP Module 2025

> Question-Answering chatbot using **Retrieval-Augmented Generation (RAG)**  
> Built with LangChain · FAISS · HuggingFace · Streamlit

**Student:** MILOUDI · **GitHub:** [Melouka-mld](https://github.com/Melouka-mld)

---

## 📌 What is RAG?

```
YOUR DOCUMENTS (data/)
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

## 📁 Project Structure

```
RAG-Chatbot-NLP_MILOUDI/
├── data/
│   └── comments.txt          ← Personal dataset (customer comments)
├── ingestion.py              ← Step 1: Load → Chunk → Embed → Store
├── retrieval.py              ← Step 2: Question → FAISS → Top-3 chunks
├── chatbot_rag.py            ← Step 3: Streamlit UI chatbot
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Framework | LangChain |
| Vector DB | FAISS (local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | google/flan-t5-large (FREE) |
| UI | Streamlit |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Melouka-mld/RAG-Chatbot-NLP_MILOUDI.git
cd RAG-Chatbot-NLP_MILOUDI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your HuggingFace token
```bash
cp .env.example .env
# Edit .env → add your token from huggingface.co/settings/tokens
```

### 4. Build the vector database
```bash
python ingestion.py
```

### 5. Launch the chatbot
```bash
streamlit run chatbot_rag.py
```

---

## 🤖 LLM Details

| Property | Value |
|----------|-------|
| Model | google/flan-t5-large |
| Architecture | T5 encoder-decoder Transformer |
| Parameters | ~780 Million |
| Training | FLAN instruction fine-tuning (Google) |
| Cost | FREE via HuggingFace Inference API |
| Link | [huggingface.co/google/flan-t5-large](https://huggingface.co/google/flan-t5-large) |

---

## 👤 Author

**MILOUDI** · NLP Module · 2025  
GitHub: [@Melouka-mld](https://github.com/Melouka-mld)
