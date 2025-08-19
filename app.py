
import os
import re
import io
import time
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import streamlit as st


try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# OpenAI is optional; we also support a local fallback via Ollama if present
OPENAI_IMPORT_ERROR = None
try:
    from openai import OpenAI
except Exception as e:
    OPENAI_IMPORT_ERROR = e
    OpenAI = None


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


def _clean_text(txt: str) -> str:
    txt = txt.replace("\x00", " ").replace("\u0000", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


SECTION_PATTERNS = [
    r"(?m)^(Section|Sec\.?)\s+(\d+(\.\d+)*)[:\.\-\s]",
    r"(?m)^(Clause)\s+(\d+(\.\d+)*)[:\.\-\s]",
    r"(?m)^(Article)\s+(\d+(\.\d+)*)[:\.\-\s]",
    r"(?m)^\d+(\.\d+)*\s+[A-Z][^\n]{0,80}$",  # numbered heading lines
    r"(?m)^[A-Z][A-Z \-]{3,}$",               # ALL CAPS headings
]


def detect_sections(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Return list of (title, (start_idx, end_idx)) for detected sections; falls back to whole doc."""
    spans = []
    # Find candidate headings
    headings = []
    for pat in SECTION_PATTERNS:
        for m in re.finditer(pat, text):
            start = m.start()
            # Heading: grab the full line
            line_start = text.rfind("\n", 0, start) + 1
            line_end = text.find("\n", start)
            if line_end == -1:
                line_end = len(text)
            title = text[line_start:line_end].strip()
            if title and (line_start, line_end, title) not in headings:
                headings.append((line_start, line_end, title))
    if not headings:
        return [("Document", (0, len(text)))]
    headings = sorted(headings, key=lambda x: x[0])
    
    for idx, (s, e, title) in enumerate(headings):
        next_start = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        spans.append((title, (s, next_start)))
    
    dedup = []
    last_end = -1
    for title, (s, e) in spans:
        if s >= last_end:
            dedup.append((title, (s, e)))
            last_end = e
    return dedup or [("Document", (0, len(text)))]


def split_into_chunks(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Simple recursive chunking along paragraph boundaries with overlap."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            # if a single paragraph is huge, hard-split
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(start + max_chars, len(p))
                    chunks.append(p[start:end])
                    start = end - overlap if end - overlap > start else end
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    
    with_overlap = []
    for i, ch in enumerate(chunks):
        if i == 0:
            with_overlap.append(ch)
        else:
            prev = chunks[i-1]
            tail = prev[-overlap:]
            merged = (tail + "\n\n" + ch).strip()
            with_overlap.append(merged)
    return with_overlap


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, Dict[int, str]]:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed")
    reader = PdfReader(io.BytesIO(file_bytes))
    text_pages = {}
    all_text = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = _clean_text(t)
        text_pages[i + 1] = t
        all_text.append(t)
    return _clean_text("\n\n".join(all_text)), text_pages


def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx2txt is None:
        raise RuntimeError("docx2txt not installed")
    tmp = io.BytesIO(file_bytes)
    
    import tempfile, shutil, zipfile
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as fp:
        fp.write(tmp.read())
        temp_path = fp.name
    try:
        txt = docx2txt.process(temp_path) or ""
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
    return _clean_text(txt)


def build_embeddings_model(name: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    return SentenceTransformer(name)


def ensure_vector_store(persist_dir: str):
    if chromadb is None:
        raise RuntimeError("chromadb not installed")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    return client


def add_to_collection(client, collection_name: str, embeddings_model, chunks: List[Chunk]):
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    # Embed in mini-batches
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [c.metadata.get("id", str(uuid.uuid4())) for c in chunks]

    batch = 64
    all_embeddings = []
    for i in range(0, len(texts), batch):
        embs = embeddings_model.encode(texts[i:i+batch], show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(embs)
    collection.add(documents=texts, embeddings=all_embeddings, metadatas=metadatas, ids=ids)
    return collection


def retrieve(client, collection_name: str, embeddings_model, query: str, top_k: int = 6) -> List[Chunk]:
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    q_emb = embeddings_model.encode([query], normalize_embeddings=True)[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("distances", [[]])[0]):
        meta = dict(meta or {})
        meta["score"] = float(1 - dist)  # cosine similarity approximation
        out.append(Chunk(text=doc, metadata=meta))
    return out


def simple_conflict_scan(chunks: List[Chunk]) -> List[Tuple[Chunk, Chunk, str]]:
    """Heuristic: flags potential conflicts when two chunks discuss same section/topic with negation mismatch."""
    conflicts = []
    def polarity(t: str) -> int:
        t_low = t.lower()
        pos = sum(1 for w in ["shall", "must", "required", "entitled"] if w in t_low)
        neg = sum(1 for w in ["shall not", "must not", "prohibited", "forbidden", "not permitted", "no liability"] if w in t_low)
        return pos - 2*neg
    for i in range(len(chunks)):
        for j in range(i+1, len(chunks)):
            a, b = chunks[i], chunks[j]
            # same topic if section titles share a keyword
            ta = (a.metadata.get("section_title") or "").lower()
            tb = (b.metadata.get("section_title") or "").lower()
            if not ta or not tb:
                continue
            shared = set(re.findall(r"[a-z]{4,}", ta)).intersection(set(re.findall(r"[a-z]{4,}", tb)))
            if not shared:
                continue
            pa, pb = polarity(a.text), polarity(b.text)
            if (pa > 0 and pb < 0) or (pa < 0 and pb > 0):
                conflicts.append((a, b, f"Potential conflict on topic: {', '.join(sorted(shared))}"))
    return conflicts


def make_citation(meta: Dict[str, Any]) -> str:
    name = meta.get("doc_name", "Document")
    section = meta.get("section_title")
    pages = meta.get("pages")
    if section and pages:
        return f"{name} — {section} (pp. {pages})"
    if section:
        return f"{name} — {section}"
    if pages:
        return f"{name} (pp. {pages})"
    return name


SYSTEM_PROMPT = """You are a meticulous legal research assistant.
You MUST answer using ONLY the provided context, citing each relevant section precisely.
If the context is insufficient or conflicting, say so clearly and list the missing info or conflicts.
Format citations inline like [Source 1], [Source 2], etc., and then provide a Source List mapping numbers to full citations.
Keep the tone crisp and neutral. Avoid definitive legal advice; present findings with references.
"""

def build_prompt(query: str, retrieved: List[Chunk]) -> Tuple[str, Dict[int, str]]:
    sources_map = {}
    context_blocks = []
    for idx, ch in enumerate(retrieved, start=1):
        sources_map[idx] = make_citation(ch.metadata)
        context_blocks.append(f"[Source {idx}] {ch.text}")
    context_text = "\n\n".join(context_blocks)
    user_prompt = f"User question:\n{query}\n\nContext:\n{context_text}\n\nInstructions: Provide a direct answer with inline citations like [Source 1], [Source 2]. After the answer, include a 'Source List' section mapping each source number to its citation."
    return user_prompt, sources_map


def call_openai(messages: List[Dict[str, str]], model: str, api_key: str, temperature: float = 0.0) -> str:
    if OpenAI is None:
        raise RuntimeError(f"openai package not available: {OPENAI_IMPORT_ERROR}")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def call_ollama(messages: List[Dict[str, str]], model: str = "llama3.1") -> str:
    """Very light client for local Ollama if running (http://localhost:11434)."""
    import json, urllib.request
    req = urllib.request.Request("http://localhost:11434/api/chat", method="POST")
    req.add_header("Content-Type", "application/json")
    payload = {"model": model, "messages": messages, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    try:
        with urllib.request.urlopen(req, data, timeout=120) as r:
            out = json.loads(r.read().decode("utf-8"))
            return out.get("message", {}).get("content", "")
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")


# -------------- Streamlit UI --------------

st.set_page_config(page_title="Multi-Document Legal RAG Assistant", page_icon="⚖️", layout="wide")
st.title("⚖️ Multi-Document Legal Research Assistant (One‑Shot RAG)")

with st.sidebar:
    st.header("Settings")
    st.markdown("**1) Upload legal documents** (PDF/DOCX).")
    files = st.file_uploader("Upload one or more files", type=["pdf", "docx"], accept_multiple_files=True)

    st.markdown("---")
    st.markdown("**2) Embeddings & Retrieval**")
    emb_model_name = st.selectbox(
        "Embedding model",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2", "mixedbread-ai/mxbai-embed-large-v1"],
        index=0
    )
    chunk_chars = st.number_input("Max chunk size (chars)", min_value=400, max_value=3000, value=1200, step=100)
    overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=500, value=150, step=10)
    top_k = st.number_input("Top-k retrieval", min_value=2, max_value=12, value=6, step=1)

    st.markdown("---")
    st.markdown("**3) Generator (LLM)**")
    gen_backend = st.selectbox("Backend", ["OpenAI", "Ollama (local)"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    openai_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0, disabled=(gen_backend != "OpenAI"))
    openai_key_input = st.text_input("OpenAI API Key (if using OpenAI)", type="password", placeholder="sk-...")
    ollama_model = st.text_input("Ollama model name (if using local)", value="llama3.1", disabled=(gen_backend != "Ollama (local)"))

    st.markdown("---")
    persist_dir = st.text_input("Vector store path", value="./rag_db")
    build_btn = st.button("🔧 Build / Refresh Index", type="primary")

# Persistent state
if "built" not in st.session_state:
    st.session_state.built = False
if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"legal-rag-{uuid.uuid4().hex[:8]}"
if "emb_model" not in st.session_state:
    st.session_state.emb_model = None
if "client" not in st.session_state:
    st.session_state.client = None

# --- Build index ---
if build_btn:
    if not files:
        st.error("Please upload at least one PDF/DOCX.")
    else:
        with st.spinner("Parsing & indexing documents..."):
            # Build embeddings
            emb_model = build_embeddings_model(emb_model_name)
            st.session_state.emb_model = emb_model
            client = ensure_vector_store(persist_dir)
            st.session_state.client = client

            all_chunks: List[Chunk] = []
            for f in files:
                name = f.name
                data = f.read()
                ext = name.lower().split(".")[-1]
                if ext == "pdf":
                    doc_text, page_map = extract_text_from_pdf(data)
                elif ext == "docx":
                    doc_text = extract_text_from_docx(data)
                    # Pages unknown for docx
                    page_map = {1: doc_text}
                else:
                    continue

                doc_text = _clean_text(doc_text)
                sections = detect_sections(doc_text)

                for title, (s, e) in sections:
                    sec_text = doc_text[s:e].strip()
                    if not sec_text:
                        continue
                    smalls = split_into_chunks(sec_text, max_chars=chunk_chars, overlap=overlap)
                    # estimate pages from occurrences (for PDFs we have page_map)
                    pages = None
                    if ext == "pdf" and page_map:
                        pages_hit = []
                        for pg, t in page_map.items():
                            if any(snippet.strip() and snippet.strip()[:60] in t for snippet in smalls[:2]):
                                pages_hit.append(pg)
                        if pages_hit:
                            pages = f"{min(pages_hit)}-{max(pages_hit)}" if len(pages_hit) > 1 else f"{pages_hit[0]}"
                    for small in smalls:
                        all_chunks.append(Chunk(
                            text=small,
                            metadata={
                                "id": str(uuid.uuid4()),
                                "doc_name": name,
                                "section_title": title[:120],
                                "pages": pages,
                            }
                        ))
            if not all_chunks:
                st.error("No text extracted from uploaded files.")
            else:
                add_to_collection(client, st.session_state.collection_name, emb_model, all_chunks)
                st.session_state.built = True
                st.success(f"Indexed {len(all_chunks)} chunks from {len(files)} file(s).")

st.markdown("---")
st.subheader("Ask a legal question")
query = st.text_input("Enter your query", placeholder="e.g., What are the termination rights and notice period?")

colA, colB = st.columns([1, 1])
with colA:
    run_btn = st.button("🔎 Retrieve & Answer", type="primary", use_container_width=True)
with colB:
    clear_btn = st.button("🗑️ Reset Session", use_container_width=True)
if clear_btn:
    st.session_state.built = False
    st.session_state.emb_model = None
    st.session_state.client = None
    st.session_state.collection_name = f"legal-rag-{uuid.uuid4().hex[:8]}"
    st.experimental_rerun()

if run_btn:
    if not st.session_state.built:
        st.error("Please build the index first.")
    elif not query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Retrieving relevant sections..."):
            retrieved = retrieve(
                st.session_state.client, st.session_state.collection_name,
                st.session_state.emb_model, query, top_k=int(top_k)
            )
        if not retrieved:
            st.warning("No relevant context found.")
        else:
            # Conflict scan (heuristic)
            conflicts = simple_conflict_scan(retrieved)
            if conflicts:
                with st.expander("⚠️ Potential conflicts detected (heuristic)"):
                    for a, b, msg in conflicts:
                        st.markdown(f"- **{msg}** between *{make_citation(a.metadata)}* and *{make_citation(b.metadata)}*")

            user_prompt, sources_map = build_prompt(query, retrieved)
            st.markdown("##### Retrieved Sources")
            for i, ch in enumerate(retrieved, start=1):
                with st.expander(f"[Source {i}] {make_citation(ch.metadata)} — score {ch.metadata.get('score', 0):.3f}"):
                    st.write(ch.text)

            st.markdown("---")
            st.subheader("Answer")

            try:
                if gen_backend == "OpenAI":
                    api_key = openai_key_input or os.getenv("OPENAI_API_KEY", "")
                    if not api_key:
                        st.error("OpenAI API key is required for the OpenAI backend.")
                        st.stop()
                    answer = call_openai(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        model=openai_model,
                        api_key=api_key,
                        temperature=temperature,
                    )
                else:
                    answer = call_ollama(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        model=ollama_model or "llama3.1",
                    )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

            # Render final with source list
            st.markdown(answer)
            st.markdown("###### Source List")
            for i, cite in sources_map.items():
                st.markdown(f"- [Source {i}] {cite}")

st.markdown("---")
with st.expander("ℹ️ How to run (one-shot)"):
    st.markdown("""
1. **Install dependencies**: `pip install -r requirements.txt`  
2. **Run app**: `streamlit run app.py`  
3. **In the sidebar**: upload PDFs/DOCX → choose embedding model → click **Build / Refresh Index**.  
4. Enter your question → **Retrieve & Answer**.  
5. Use **OpenAI** (set API key) or **Ollama** (local model) for generation.

**Notes**
- Citations are inserted inline like `[Source 2]` and mapped below to *document — section — pages*.
- Conflict detection is heuristic; verify critical findings.
- Vector store persists at the chosen path so you can reuse across sessions.
""")
