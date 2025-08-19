---

# ⚖️ ***Multi-Document Legal RAG Assistant***

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A ***Retrieval-Augmented Generation (RAG)*** powered assistant for ***legal document research & analysis***.
Easily upload **contracts, statutes, or case law documents**, ask questions in plain English, and receive ***contextual answers with legal citations & conflict detection***.

---

## ✨ ***Features***

* 📂 ***Multi-format support*** → Upload **PDF/DOCX** legal documents
* 🏛 ***Legal-aware structure*** → Detects *Sections, Clauses, Articles, Headings*
* 🔎 ***Smart retrieval*** → Finds relevant sections using **embeddings**
* 📖 ***Inline citations*** → Answers with proper **source mapping**
* ⚠️ ***Conflict detection*** → Flags contradictions between documents
* 🧠 ***Flexible LLMs*** → Use **OpenAI GPT** (cloud) or **Ollama LLaMA** (local, free)

---

## 🛠 ***Tech Stack***

* **Frontend:** Streamlit
* **Vector Store:** ChromaDB
* **Embeddings:** SentenceTransformers (*all-MiniLM-L6-v2*)
* **LLMs:**

  * OpenAI (`gpt-4o`, `gpt-4.1-mini`)
  * Ollama (`llama3.1`)

---

## 📂 ***Project Structure***

```
├── app.py              # Streamlit application
├── requirements.txt    # Dependencies
├── rag_db/             # Vector database storage
└── README.md           # Documentation
```

---

## ⚙️ ***Installation***

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/legal-rag-assistant.git
cd legal-rag-assistant
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run App

```bash
streamlit run app.py
```

👉 App will run locally at: ***https://bhavyadashottar18-multi-document-legal-research-83ff8ff.hf.space/***

---

## ▶️ ***Usage Guide***

1. 📂 Upload one or more **legal documents (PDF/DOCX)**
2. 🔧 Click **“Build / Refresh Index”**
3. ❓ Ask a **legal question** in plain English
4. 📑 View:

   * Retrieved **legal clauses**
   * AI-generated **answer with citations**
   * ⚠️ Detected **conflicts** (if any)

---

## 🔍 ***Example Query***

**Question:**

> What are the termination rights in the agreement?

**AI Answer:**
Termination requires a 30-day written notice ***\[Source 2]***.

**Source List:**

* ***\[Source 2]*** Employment Agreement — Clause 7: Termination (pp. 4–5)

---

## ⚠️ ***Limitations***

* 🔑 **OpenAI backend** → Requires API key
* 🖥 **Ollama backend** → Works only on local systems (not Colab)
* ⚖️ **Disclaimer** → This tool aids research but is ***not legal advice***

---

## 🚀 ***Future Roadmap***

* 📊 Visual clause analytics (graphs, stats)
* 🌐 Support for **TXT, HTML, scanned OCR PDFs**
* 🤖 Stronger NLP-based ***conflict resolution***
* 🗂 Integration with **external legal databases**

---

## 👨‍💻 ***Author***

Developed by ***Bhavya Dashottar*** ✨
*AI/ML Engineer | LegalTech Enthusiast*

---
