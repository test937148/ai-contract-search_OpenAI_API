import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import re
import pandas as pd
import json
from hashlib import md5
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

st.set_page_config(page_title="AI Contract Search (GPT-5)", layout="wide")
st.title("📄 AI Contract Search (GPT-5)")

# --------------------
# 1. Load embedding model
# --------------------
@st.cache_resource
def load_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_models()

# --------------------
# 2. Initialize OpenAI GPT-5 client
# --------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("⚠️ OpenAI API key not found! Please add OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --------------------
# 3. Utility functions
# --------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def chunk_text(text, max_words=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start += max_words - overlap
    return chunks

def extract_paragraph(text, answer):
    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        if answer.lower() in para.lower():
            return para.strip()
    return text[:500] + "..."

def hash_file(filename, content_bytes):
    h = md5()
    h.update(filename.encode('utf-8'))
    h.update(content_bytes)
    return h.hexdigest()

# --------------------
# 4. Upload PDFs and create chunks
# --------------------
uploaded_files = st.file_uploader("Upload contract PDFs", type=["pdf"], accept_multiple_files=True)
chunks_data = []

cache_file = "embeddings_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        cache = json.load(f)
else:
    cache = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        file_hash = hash_file(uploaded_file.name, file_bytes)

        if file_hash in cache:
            chunks_data.extend(cache[file_hash])
            continue

        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            file_chunks = []
            for page_num, page in enumerate(pdf, start=1):
                page_text = page.get_text()
                for idx, chunk in enumerate(chunk_text(page_text)):
                    embedding = embed_model.encode(chunk).tolist()
                    chunk_obj = {
                        "filename": uploaded_file.name,
                        "page": page_num,
                        "chunk_id": idx + 1,
                        "text": chunk,
                        "embedding": embedding
                    }
                    file_chunks.append(chunk_obj)
                    chunks_data.append(chunk_obj)
        cache[file_hash] = file_chunks
        with open(cache_file, "w") as f:
            json.dump(cache, f)
    st.success(f"✅ Loaded {len(uploaded_files)} documents and {len(chunks_data)} chunks.")

# --------------------
# 5. Section classification (optional)
# --------------------
def classify_section(chunk_text):
    prompt = f"Classify this contract text into a standard section (Termination, Payment, Confidentiality, Liability, Misc). Only return the section name.\n\nText:\n{chunk_text[:500]}"
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role":"user","content":prompt}]
        )
        return response.choices[0].message.content.strip()
    except:
        return "Unknown"

for chunk in chunks_data:
    if "section" not in chunk:
        chunk["section"] = classify_section(chunk["text"])

# --------------------
# 6. Semantic search
# --------------------
def search(query, chunks, top_n=5, section_filter=None):
    query_emb = embed_model.encode(query)
    sims = []
    for chunk in chunks:
        if section_filter and chunk.get("section") != section_filter:
            continue
        sim = cosine_similarity(query_emb, np.array(chunk['embedding']))
        sims.append((chunk, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]

def answer_with_gpt(question, top_chunks, max_chunks_for_gpt=3):
    selected_chunks = top_chunks[:max_chunks_for_gpt]
    context = "\n\n".join([c['text'] for c, _ in selected_chunks])
    prompt = f"You are a legal assistant. Answer the question ONLY using the context below.\n\nQuestion: {question}\n\nContext:\n{context}"
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

# --------------------
# 7. User input
# --------------------
query = st.text_input("Enter your question:")
section_options = list(set([c.get("section","Unknown") for c in chunks_data]))
section_filter = st.selectbox("Filter by contract section (optional)", ["All"] + section_options)

# --------------------
# 8. Search & Display results
# --------------------
if query and chunks_data:
    section_filter_value = None if section_filter=="All" else section_filter
    top_chunks = search(query, chunks_data, top_n=10, section_filter=section_filter_value)

    st.subheader("🔍 Best Matches")
    report_rows = []

    gpt_answer = answer_with_gpt(query, top_chunks, max_chunks_for_gpt=3)

    for chunk, score in top_chunks[:5]:
        paragraph = extract_paragraph(chunk['text'], gpt_answer)
        st.markdown(
            f"**📄 {chunk['filename']}** - Page {chunk['page']} - Chunk {chunk['chunk_id']} "
            f"- Section: {chunk.get('section','Unknown')} (Similarity: {score:.3f})"
        )
        st.markdown(f"**Answer:** {gpt_answer}")
        st.markdown(f"**Context Paragraph:**\n> {paragraph}")
        st.markdown("---")

        report_rows.append({
            "filename": chunk['filename'],
            "page": chunk['page'],
            "chunk_id": chunk['chunk_id'],
            "section": chunk.get("section", "Unknown"),
            "similarity": score,
            "answer": gpt_answer,
            "context": paragraph
        })

    df_report = pd.DataFrame(report_rows)
    csv_data = df_report.to_csv(index=False)
    st.download_button(
        label="💾 Download Full Search Report as CSV",
        data=csv_data,
        file_name="contract_search_report.csv",
        mime="text/csv"
    )
