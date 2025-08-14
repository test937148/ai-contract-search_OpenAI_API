import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import re
import pandas as pd
import os
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from hashlib import md5

st.set_page_config(page_title="AI Contract Search (GPT-5)", layout="wide")
st.title("📄 AI-Powered Contract Search (GPT-5)")

# --------------------
# 1. Load models
# --------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embed_model

embed_model = load_models()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --------------------
# 2. Utilities
# --------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def chunk_text(text, max_words=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks

def extract_paragraph(text, answer):
    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        if answer.lower() in para.lower():
            return para.strip()
    return text[:500] + "..."

def hash_file(filename, content):
    h = md5()
    h.update(filename.encode('utf-8'))
    h.update(content.encode('utf-8'))
    return h.hexdigest()

# --------------------
# 3. Upload PDFs
# --------------------
uploaded_files = st.file_uploader("Upload your contract PDFs", type=["pdf"], accept_multiple_files=True)
chunks_data = []

cache_file = "embeddings_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        cache = json.load(f)
else:
    cache = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read().decode("latin1")  # fallback for PDF bytes
        file_hash = hash_file(uploaded_file.name, file_content)
        
        if file_hash in cache:
            # Load cached embeddings
            chunks_data.extend(cache[file_hash])
        else:
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                file_chunks = []
                for page_num, page in enumerate(pdf, start=1):
                    page_text = page.get_text()
                    page_chunks = chunk_text(page_text, max_words=500, overlap=50)
                    for idx, chunk in enumerate(page_chunks):
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
                # Cache
                cache[file_hash] = file_chunks
                with open(cache_file, "w") as f:
                    json.dump(cache, f)
    st.success(f"✅ Loaded {len(uploaded_files)} documents and {len(chunks_data)} chunks.")

# --------------------
# 4. Section Classification
# --------------------
def classify_section(chunk_text):
    prompt = f"Classify this contract text into a standard section (e.g., Termination, Payment, Confidentiality, Liability, Misc). Only return the section name.\n\nText:\n{chunk_text[:500]}"
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role":"user","content":prompt}]
    )
    section = response.choices[0].message.content.strip()
    return section

for chunk in chunks_data:
    if "section" not in chunk:
        try:
            chunk["section"] = classify_section(chunk["text"])
        except:
            chunk["section"] = "Unknown"

# --------------------
# 5. Semantic Search
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
# 6. User Input
# --------------------
query = st.text_input("Enter your question:")
section_options = list(set([c.get("section", "Unknown") for c in chunks_data]))
section_filter = st.selectbox("Filter by contract section (optional)", ["All"] + section_options)

# --------------------
# 7. Run Search & Display Results
# --------------------
if query and chunks_data:
    section_filter_value = None if section_filter=="All" else section_filter
    top_chunks = search(query, chunks_data, top_n=10, section_filter=section_filter_value)

    st.subheader("🔍 Best Matches")
    report_rows = []

    gpt_answer = answer_with_gpt(query, top_chunks, max_chunks_for_gpt=3)

    for chunk, score in top_chunks[:5]:  # top 5 chunks UI display
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
