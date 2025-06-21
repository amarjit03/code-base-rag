# app.py
import os
import shutil
import zipfile
import streamlit as st
from embedding_builder_cohere import build_embedding_index
from utils import create_or_load_index, get_llm_response

st.set_page_config(page_title="🧠 RAG Code Assistant", page_icon="🧠")
st.title("🧠 RAG-based Code Assistant")

uploaded_file = st.file_uploader("📁 Upload your zipped Python codebase", type=["zip"])
query = st.text_input("💬 Ask a question about your codebase:")

CODEBASE_PATH = "codebase"

if uploaded_file:
    if os.path.exists(CODEBASE_PATH):
        shutil.rmtree(CODEBASE_PATH)
    os.makedirs(CODEBASE_PATH, exist_ok=True)
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        zip_ref.extractall(CODEBASE_PATH)
    st.success("✅ Codebase extracted successfully.")
    build_embedding_index()

if st.button("Get Answer") and query:
    try:
        with st.spinner("🔍 Thinking..."):
            db = create_or_load_index()
            response = get_llm_response(query, db)
        st.success("✅ Answer:")
        st.markdown(f"```\n{response}\n```")
    except Exception as e:
        st.error(f"❌ Error: {e}")
