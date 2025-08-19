⚖️ Multi-Document Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) system designed for legal document research.
The app allows users to upload contracts, statutes, and case law (PDF/DOCX), ask natural questions, and receive answers grounded in the documents with citations.

🚀 Features

Multi-format support → Upload PDF and DOCX legal documents.

Document parsing → Extracts, cleans, and structures text into sections (Clauses, Articles, etc.).

Semantic retrieval → Embeddings with sentence-transformers ensure context-aware search.

Citations → Answers include inline citations [Source 1] with a Source List mapping to section/page references.

Conflict detection → Flags contradictory clauses across multiple documents.

Hierarchical organization → Each chunk linked to Document → Section → Pages.

LLM backends →

OpenAI (cloud) → GPT-4 family (requires API key).

Ollama (local) → Free local inference with models like llama3.1.

🛠️ Tech Stack

Frontend: Streamlit

Vector Database: ChromaDB

Embeddings: SentenceTransformers

LLMs:

OpenAI (gpt-4o, gpt-4.1-mini, etc.)

Ollama (local, e.g., llama3.1)

📂 Project Structure
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
└── rag_db/             # Persistent vector store (auto-created)

⚙️ Installation & Setup
1️⃣ Clone and install dependencies
git clone <repo-url>
cd legal-rag-assistant
pip install -r requirements.txt

2️⃣ Run the app locally
streamlit run app.py


App will be available at: http://localhost:8501

3️⃣ In Google Colab (for demo)
!pip install streamlit pyngrok cloudflared chromadb pypdf docx2txt sentence-transformers openai


Then launch with cloudflared or ngrok for a public URL.

🔑 Usage Instructions

Upload legal documents (PDF/DOCX) in the sidebar.

Build Index → Documents are chunked, embedded, and stored in ChromaDB.

Ask a Question in natural language (e.g., “What is the deposit amount?”).

Retrieve & Answer →

Relevant clauses are displayed with scores.

The answer is generated with citations.

Conflicting clauses are flagged if detected.

📌 Example Query

Q: What is the termination clause in this agreement?
Answer (sample):
Termination rights are provided under Clause 7, requiring a 30-day prior notice [Source 2].

Source List

[Source 2] License Agreement — Clause 7 (pp. 4-5)

⚠️ Notes & Limitations

OpenAI backend requires an API key (sk-...).

Ollama backend works only on a local machine, not in Colab.

In Colab demo, only the retrieval pipeline can be shown unless an OpenAI key is provided.

This system is a legal research assistant, not a replacement for professional legal advice.

👨‍💻 Author

Developed as part of an assignment project by Bhavya Dashottar.
