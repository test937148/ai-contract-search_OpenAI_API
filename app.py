# app.py — Streamlit + OpenAI GPT-5 contract Q&A with semantic retrieval

import os
import re
import hashlib
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from typing import List, Dict, Tuple

# --- UI setup ---
st.set_page_config(page_title="AI Contract Search (GPT-5)", page_icon="📄", layout="wide")
st.title("📄 AI Contract Search & Q&A (GPT-5)")
st.caption("Upload PDFs → semantic retrieve with OpenAI embeddings → answer with GPT-5.")

# --- Sidebar: API key & settings ---
st.sidebar.header("Settings")
# Prefer Streamlit secrets in production; allow manual entry for local/testing
api_key = st.secrets.get("OPENAI_API_KEY", None)
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=api_key if api_key else "",
    type="password",
    help="Store this in Streamlit Cloud → Settings → Secrets as OPENAI_API_KEY for production."
)

# Model selectors
GPT_MODEL = st.sidebar.selectbox("Answering model", ["gpt-5", "o3-pro", "gpt-4o-mini"], index=0)
EMBED_MODEL = st.sidebar.selectbox(
    "Embedding model",
    ["text-embedding-3-small", "text-embedding-3-large"],
    index=0,
    help="Small = cheaper/faster. Large = higher recall/precision."
)

# Retrieval tuning
chunk_chars = st.sidebar.slider("Chunk size (characters)", 400, 2000, 900, 50)
overlap_chars = st.sidebar.slider("Overlap (characters)", 0, 400, 120, 10)
top_k_candidates = st.sidebar.slider("Candidate chunks (pre-rerank)", 3, 30, 12, 1)
top_k_final = st.sidebar.slider("Final chunks to send to GPT", 1, 8, 4, 1)
keyword_boost = st.sidebar.slider("Keyword boost (per term)", 0.00, 0.20, 0.05, 0.01)

# --- Imports that depend on API key / SDK ---
if api_key:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.stop()
else:
    st.info("Enter your OpenAI API key in the sidebar to enable GPT-5 answers and embeddings.")
    client = None

# --- Helpers ---
def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def chunk_text_pages(pages: List[Tuple[int, str]], chunk_size: int, overlap: int) -> List[Dict]:
    """
    Build overlapping character-based chunks across pages while preserving file/page refs.
    pages: list of (page_number, page_text)
    Returns list of dicts with: text, page, span_start, span_end
    """
    chunks = []
    buf = ""
    meta = []
    for page_no, page_text in pages:
        # Normalize whitespace a bit
        page_text = re.sub(r'\n{3,}', '\n\n', page_text.strip())
        if not page_text:
            continue
        # Build chunks by chars
        idx = 0
        while idx < len(page_text):
            take = page_text[idx: idx + (chunk_size if chunk_size > 0 else 1000)]
            if take:
                chunks.append({
                    "text": take,
                    "page": page_no + 1,  # 1-based for display
                })
            idx += max(1, chunk_size - overlap) if chunk_size else 1000
    return chunks

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A: (n, d), b: (d,)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return A_norm @ b_norm

@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: List[str], embed_model: str, api_key_snapshot: str) -> List[List[float]]:
    """
    Caches embeddings by (texts, model, api key snapshot length).
    We don't store keys; just use its length to avoid collisions if you switch orgs/keys.
    """
    embeddings = []
    # Batch to reduce round-trips
    BATCH = 96
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=embed_model, input=batch)
        # Order preserved
        for item in resp.data:
            embeddings.append(item.embedding)
    return embeddings

def hybrid_score(sim: float, text: str, query_terms: List[str], boost_per_term: float) -> float:
    if boost_per_term <= 0:
        return sim
    t = text.lower()
    extra = sum(1 for w in query_terms if w and w in t) * boost_per_term
    return sim + extra

def build_gpt_prompt(question: str, hits: List[Dict]) -> List[Dict]:
    """
    Create a strict, grounded prompt for GPT-5 with explicit instructions.
    """
    context_blocks = []
    for h in hits:
        header = f"[{h['filename']} — page {h['page']} — score {h['score']:.3f}]"
        context_blocks.append(header + "\n" + h['text'].strip())

    context_text = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a contract analyst. Answer ONLY from the provided context. "
        "If the answer is not present, say so clearly. Quote exact clauses when helpful, "
        "and reference the file name and page number in parentheses."
    )
    user = (
        f"Question: {question}\n\n"
        f"Context (multiple excerpts):\n{context_text}\n\n"
        "Return:\n- A concise answer first line.\n- Then a short explanation with quoted phrases.\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# --- App body ---
uploaded_files = st.file_uploader("Upload contract PDFs", type=["pdf"], accept_multiple_files=True)

documents = []
if uploaded_files:
    all_chunks = []
    for i, uf in enumerate(uploaded_files, start=1):
        # Read PDF bytes once per file
        data = uf.read()
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for p in doc:
            pages.append((p.number, p.get_text()))
        chunks = chunk_text_pages(pages, chunk_chars, overlap_chars)
        # Attach filename
        for c in chunks:
            c["filename"] = uf.name
        all_chunks.extend(chunks)

    st.success(f"Prepared {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")

    # Build embeddings
    if client is None:
        st.stop()

    st.write("Embedding chunks…")
    texts = [c["text"] for c in all_chunks]
    try:
        vecs = embed_texts_cached(texts, EMBED_MODEL, f"klen:{len(api_key)}|model:{EMBED_MODEL}")
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        st.stop()

    # Store as numpy for fast math
    M = np.array(vecs, dtype=np.float32)

    # Search controls
    st.divider()
    st.subheader("Ask a question")
    query = st.text_input("Your question about the uploaded contracts")
    if query:
        # Embed query
        try:
            q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
        except Exception as e:
            st.error(f"Query embedding failed: {e}")
            st.stop()

        sims = cosine_sim_matrix(M, np.array(q_emb, dtype=np.float32))
        # Hybrid scoring (semantic + keyword presence)
        q_terms = re.findall(r"[a-zA-Z0-9\-]{3,}", query.lower())
        scores = []
        for idx, s in enumerate(sims.tolist()):
            boosted = hybrid_score(s, all_chunks[idx]["text"], q_terms, keyword_boost)
            scores.append((idx, boosted, s))

        # Take candidate set, sort by boosted score
        scores.sort(key=lambda x: x[1], reverse=True)
        candidate_idxs = [i for (i, _, _) in scores[:top_k_candidates]]

        # Build candidate list with metadata
        candidates = []
        for i in candidate_idxs:
            c = dict(all_chunks[i])
            c["cosine"] = float(sims[i])
            # temporary boosted score stored as 'score' for display
            c["score"] = float([b for (idx, b, s) in scores if idx == i][0])
            candidates.append(c)

        # Narrow to final K by semantic cosine (tie-breaker)
        candidates.sort(key=lambda x: (x["score"], x["cosine"]), reverse=True)
        final_hits = candidates[:top_k_final]

        # Show retrieved contexts
        with st.expander("Show retrieved context chunks"):
            for h in final_hits:
                st.markdown(
                    f"**{h['filename']} — page {h['page']}**  \n"
                    f"Boosted score: {h['score']:.3f} | Cosine: {h['cosine']:.3f}"
                )
                st.code(h["text"])

        # Build GPT prompt
        messages = build_gpt_prompt(query, final_hits)

        st.write("Asking GPT…")
        try:
            # Use Chat Completions (widely supported); you can switch to Responses API if you prefer.
            completion = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0.2
            )
            answer = completion.choices[0].message.content
            st.subheader("Answer")
            st.markdown(answer)

        except Exception as e:
            # Friendly fallback suggestion
            st.error(
                "OpenAI call failed. Possible causes: invalid key, quota limits, or model access.\n\n"
                f"Details: {e}"
            )
