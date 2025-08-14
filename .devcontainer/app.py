import os
import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# Load API key from Streamlit secrets
api_key = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

st.title("📄 AI Contract Search (GPT-5, Multiple PDFs)")

uploaded_files = st.file_uploader(
    "Upload one or more PDF contracts",
    type=["pdf"],
    accept_multiple_files=True
)

question = st.text_input("Ask a question about the uploaded contracts:")

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

if uploaded_files and question:
    results = []

    for file in uploaded_files:
        pdf_text = extract_text_from_pdf(file.read())

        prompt = f"""
        You are an AI contract analyst.
        You will read the following contract and answer the user's question truthfully.

        Contract text:
        {pdf_text}

        Question: {question}

        If the answer is not explicitly stated in the contract, say "Not found in this document."
        """

        try:
            response = client.chat.completions.create(
                model="gpt-5",  # GPT-5
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content.strip()
            results.append((file.name, answer))

        except Exception as e:
            results.append((file.name, f"Error: {e}"))

    st.subheader("Search Results")
    for filename, answer in results:
        st.markdown(f"**📄 {filename}:** {answer}")
