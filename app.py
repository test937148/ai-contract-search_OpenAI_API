# requirements.txt
# streamlit
# openai
# sentence-transformers
# PyMuPDF

import streamlit as st
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

# Load the embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("📄 AI Contract Search (GPT-5 Enhanced)")
st.write("Upload PDFs and ask questions — the AI will search and answer from the documents.")

# OpenAI API Key input
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# File uploader
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

documents = []

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files, start=1):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
            documents.append({
                "id": i,
                "filename": uploaded_file.name,
                "text": text.strip(),
                "embedding": embed_model.encode(text.strip())
            })
    st.success(f"Uploaded {len(documents)} document(s).")

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Search function
def semantic_search(query, docs, top_n=1):
    query_emb = embed_model.encode(query)
    sims = []
    for doc in docs:
        sim = cosine_similarity(query_emb, doc["embedding"])
        sims.append((doc, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]

# GPT-5 answer function
def gpt5_answer(api_key, question, context):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5",  # Latest GPT model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided document context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        # Removed temperature to avoid error
    )
    return response.choices[0].message.content.strip()

# Question input
if documents and openai_api_key:
    question = st.text_input("Ask a question about your documents")
    if st.button("Search and Answer"):
        results = semantic_search(question, documents, top_n=1)
        top_doc, score = results[0]
        answer = gpt5_answer(openai_api_key, question, top_doc["text"])
        st.subheader("Answer")
        st.write(answer)
        st.subheader(f"From: {top_doc['filename']} (Score: {score:.3f})")
