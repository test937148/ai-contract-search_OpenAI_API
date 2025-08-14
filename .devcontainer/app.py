import os
import json
import re
from hashlib import md5

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="AI Contract Search — Per-Document Answers", layout="wide")
st.title("📄 AI Contract Search — Per-Document, Grounded Answers")

# -------------------------------
# Secrets / API key & model
# -------------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("⚠️ OpenAI API key is missing. Add it in Streamlit Secrets as TOML:\n\nOPENAI_API_KEY = \"sk-...\"")
    st.stop()

OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4.1")  # set to "gpt-5" if you have access
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------
# Load embedding model (cached)
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedder()

# -------------------------------
# Utilities
# -------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def chunk_text(text: str, max_words: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start += max_words - overlap
    return chunks

def hash_bytes_for_cache(filename: str, content_bytes: bytes) -> str:
    h = md5()
    h.update(filename.encode("utf-8"))
    h.update(content_bytes)
    return h.hexdigest()

def extract_best_paragraph(text: str, answer_hint: str) -> str:
    # try to surface the most relevant paragraph that contains a key phrase of the answer
    if not answer_hint:
        return text[:600] + "..." if len(text) > 600 else text
    paragraphs = re.split(r"\n\s*\n", text)
    hint = answer_hint.strip().lower()
    for p in paragraphs:
        if hint and hint in p.lower():
            return p.strip()
    return text[:600] + "..." if len(text) > 600 else text

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# -------------------------------
# Embedding Cache (JSON file)
# -------------------------------
CACHE_PATH = "embeddings_cache.json"
if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r") as f:
            EMB_CACHE = json.load(f)
    except Exception:
        EMB_CACHE = {}
else:
    EMB_CACHE = {}

def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(EMB_CACHE, f)

# -------------------------------
# Upload & preprocess PDFs
# -------------------------------
uploaded_files = st.file_uploader("Upload one or more contract PDFs", type=["pdf"], accept_multiple_files=True)

# Each chunk will be: {filename, page, chunk_id, text, embedding}
all_chunks = []

if uploaded_files:
    for f in uploaded_files:
        # Read once, reuse bytes (prevents EmptyFileError)
        file_bytes = f.read()
        file_hash = hash_bytes_for_cache(f.name, file_bytes)

        if file_hash in EMB_CACHE:
            # Use cached chunks
            all_chunks.extend(EMB_CACHE[file_hash])
            continue

        # Extract & chunk per page (so we keep page provenance)
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            file_chunks = []
            for page_num, page in enumerate(pdf, start=1):
                page_text = page.get_text() or ""
                if not page_text.strip():
                    continue
                page_chunks = chunk_text(page_text, max_words=500, overlap=50)
                for idx, chunk in enumerate(page_chunks, start=1):
                    emb = embed_model.encode(chunk).tolist()
                    file_chunks.append({
                        "filename": f.name,
                        "page": page_num,
                        "chunk_id": idx,
                        "text": chunk,
                        "embedding": emb
                    })

        # Save to in-memory and cache
        EMB_CACHE[file_hash] = file_chunks
        save_cache()
        all_chunks.extend(file_chunks)

    st.success(f"✅ Loaded {len(uploaded_files)} PDF(s) → {len(all_chunks)} chunks indexed.")

# -------------------------------
# Search helpers
# -------------------------------
def search_per_document(query: str, chunks: list, top_k_per_doc: int = 3):
    """
    Returns a dict: filename -> [(chunk_dict, similarity_score), ...] (top_k_per_doc per file)
    """
    if not chunks:
        return {}

    q_emb = embed_model.encode(query)
    per_doc_scores = {}

    # Group chunks by filename
    by_file = {}
    for c in chunks:
        by_file.setdefault(c["filename"], []).append(c)

    for filename, c_list in by_file.items():
        scored = []
        for c in c_list:
            sim = cosine_similarity(q_emb, np.array(c["embedding"], dtype=np.float32))
            scored.append((c, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        per_doc_scores[filename] = scored[:top_k_per_doc]

    return per_doc_scores

def ask_gpt_grounded(question: str, chunks_for_doc: list):
    """
    Ask GPT with only this document's top chunks.
    Forces grounded behavior and returns a structured result.
    """
    if not chunks_for_doc:
        return {
            "answer": "NOT_FOUND",
            "explanation": "No relevant passages found.",
            "support": [],
            "confidence": "low"
        }

    # Build compact context with provenance markers
    context_blocks = []
    support_meta = []
    for c, score in chunks_for_doc:
        context_blocks.append(f"[{c['filename']} | page {c['page']} | chunk {c['chunk_id']} | sim {score:.3f}]\n{c['text']}")
        support_meta.append({"filename": c["filename"], "page": c["page"], "chunk_id": c["chunk_id"], "similarity": float(score)})

    context = "\n\n".join(context_blocks)

    # Strict grounding instructions + JSON output
    system_msg = (
        "You are a contract analysis assistant. Answer ONLY using the provided context. "
        "If the answer is not explicitly present, reply with JSON where 'answer' is 'NOT_FOUND'. "
        "Be concise and do not infer beyond the text."
    )
    user_prompt = (
        "Return a compact JSON object with keys: answer (string), explanation (string), "
        "confidence (low|medium|high).\n"
        "Question: " + question + "\n\n"
        "Context:\n" + context
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ]
            # Do NOT set temperature to avoid model-specific constraints
        )
        content = resp.choices[0].message.content
    except Exception as e:
        return {
            "answer": "ERROR",
            "explanation": f"OpenAI call failed: {e}",
            "support": support_meta,
            "confidence": "low"
        }

    # Best-effort JSON parse
    import json as pyjson
    parsed = None
    try:
        parsed = pyjson.loads(content)
    except Exception:
        # fallback: wrap as free text
        parsed = {"answer": content.strip(), "explanation": "", "confidence": "low"}

    # Attach support meta so we can show page provenance
    parsed["support"] = support_meta
    return parsed

# -------------------------------
# UI Controls
# -------------------------------
st.markdown("Ask a question and get **separate answers for each document**, grounded in that document’s text.")
query = st.text_input("🔎 Your question (e.g., 'What is the termination date?')", "")
col1, col2 = st.columns(2)
with col1:
    top_k_per_doc = st.slider("Chunks per document sent to GPT", 1, 8, 3)
with col2:
    show_csv = st.checkbox("Enable CSV report download", value=True)

# -------------------------------
# Run search & show per-document answers
# -------------------------------
if query and all_chunks:
    per_doc_matches = search_per_document(query, all_chunks, top_k_per_doc=top_k_per_doc)

    st.subheader("📚 Answers by Document")

    report_rows = []
    for filename, top_matches in per_doc_matches.items():
        result = ask_gpt_grounded(query, top_matches)

        # Prepare display
        answer = result.get("answer", "").strip()
        explanation = result.get("explanation", "").strip()
        confidence = result.get("confidence", "low")

        # Pick one representative paragraph (from the best chunk) to display
        best_chunk, best_sim = (top_matches[0] if top_matches else (None, 0.0))
        paragraph = ""
        if best_chunk:
            # Try to highlight / surface a relevant paragraph using the answer as a hint
            paragraph = extract_best_paragraph(best_chunk["text"], answer if answer not in ("NOT_FOUND", "ERROR") else "")

        # Render card for this file
        st.markdown(f"### 📄 {filename}")
        if answer == "NOT_FOUND":
            st.info("Couldn’t find this information in this document.")
        elif answer == "ERROR":
            st.error(f"AI error: {explanation}")
        else:
            st.markdown(f"**Answer:** {answer}")
            if explanation:
                st.caption(f"Reasoning (from context): {explanation}")
            st.caption(f"Confidence: {confidence}")

        # Show supporting snippet & provenance
        if paragraph:
            st.markdown("**Context snippet (from the most similar chunk):**")
            st.write("> " + paragraph.replace("\n", "\n> "))

        # Show top matches metadata
        with st.expander("Show supporting chunks & scores"):
            for c, s in top_matches:
                st.markdown(f"- Page **{c['page']}**, Chunk **{c['chunk_id']}**, Similarity **{s:.3f}**")

        st.markdown("---")

        # Add to report rows
        # Include first support item’s page for convenience
        first_supp_page = result.get("support", [{}])[0].get("page") if result.get("support") else None
        report_rows.append({
            "filename": filename,
            "answer": answer,
            "confidence": confidence,
            "best_chunk_page": first_supp_page,
            "top_chunk_similarity": safe_float(best_sim),
            "snippet": paragraph
        })

    # CSV download
    if show_csv and report_rows:
        df = pd.DataFrame(report_rows)
        st.download_button(
            label="💾 Download per-document answers (CSV)",
            data=df.to_csv(index=False),
            file_name="contract_answers_per_document.csv",
            mime="text/csv"
        )

elif query and not all_chunks:
    st.warning("Upload at least one PDF to search.")
