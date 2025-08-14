import os
import fitz  # PyMuPDF
import numpy as np
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ----------------------
# STREAMLIT UI SETUP
# ----------------------
st.set_page_config(page_title="AI Contract Search", page_icon="📄", layout="wide")
st.title("📄 AI Contract Search with GPT-5")
st.write("Upload your PDF contracts and ask questions — powered by semantic search + GPT-5.")

# ----------------------
# OPENAI API KEY
# ----------------------
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ----------------------
# FILE UPLOAD
# ----------------------
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    for i, file in enumerate(uploaded_files, start=1):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        documents.append({
            "id": i,
            "filename": file.name,
            "text": text.strip(),
            "embedding": embed_model.encode(text.strip())
        })

    # ----------------------
    # SEARCH FUNCTION
    # ----------------------
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

    # ----------------------
    # USER QUESTION
    # ----------------------
    query = st.text_input("Ask a question about your contracts:")

    if query:
        results = search(query, documents, top_n=1)
        for doc, score in results:
            st.subheader(f"Best Match: {doc['filename']} (score {score:.3f})")

            # Send context + question to GPT-5
            prompt = f"""
            You are an AI assistant specialized in contract review.
            Context from the document:
            {doc['text'][:4000]}

            Question: {query}

            Please give a clear and precise answer based only on the above context.
            """

            try:
                completion = client.chat.completions.create(
                    model="gpt-5",  # Latest GPT-5 model
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = response.choices[0].message.content
                st.write("**Answer:**", answer)

            except Exception as e:
                st.error(f"OpenAI call failed: {e}")

