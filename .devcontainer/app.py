import os
import fitz  # PyMuPDF
import numpy as np
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --------------------------
# 1. Streamlit App Setup
# --------------------------
st.set_page_config(page_title="AI Contract Search", layout="wide")
st.title("📄 AI-Powered Contract Search (GPT-5 + Semantic Search)")

# --------------------------
# 2. OpenAI API Key
# --------------------------
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to start.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# --------------------------
# 3. File Upload
# --------------------------
uploaded_files = st.file_uploader("Upload contract PDFs", type=["pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload at least one PDF contract to search.")
    st.stop()

# --------------------------
# 4. Extract Text from PDFs
# --------------------------
documents = []
for i, uploaded_file in enumerate(uploaded_files, start=1):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    documents.append({"id": i, "filename": uploaded_file.name, "text": text.strip()})

# --------------------------
# 5. Semantic Embeddings
# --------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

for doc in documents:
    doc['embedding'] = embed_model.encode(doc['text'])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, docs, top_n=1):
    query_emb = embed_model.encode(query)
    sims = []
    for doc in docs:
        sim = cosine_similarity(query_emb, doc['embedding'])
        sims.append((doc, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]

def extract_paragraph(text, answer):
    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        if answer.lower() in para.lower():
            return para.strip()
    return text[:500] + "..."

# --------------------------
# 6. Question Input
# --------------------------
query = st.text_input("Ask a question about the contracts:")
if not query:
    st.stop()

# --------------------------
# 7. Search & Answer
# --------------------------
results = search(query, documents, top_n=1)

for doc, score in results:
    context_text = doc['text']

    # Call GPT-5 with context
    try:
        response = client.chat.completions.create(
            model="gpt-5",  # Use latest GPT-5
            messages=[
                {"role": "system", "content": "You are a legal assistant that answers based on provided contract text."},
                {"role": "user", "content": f"Question: {query}\n\nContract:\n{context_text}"}
            ]
        )
        answer = response.choices[0].message.content
        paragraph = extract_paragraph(context_text, answer)

        st.subheader(f"Best match: {doc['filename']} (score: {score:.3f})")
        st.markdown(f"**Answer:** {answer}")
        st.markdown(f"**Context paragraph:**\n\n{paragraph}")

    except Exception as e:
        st.error(f"OpenAI call failed: {e}")
