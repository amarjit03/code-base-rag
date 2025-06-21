#  CodeBase-RAG

Welcome to **CodeBase-RAG**, a lightweight Retrieval-Augmented Generation (RAG) powered app that helps you **search and understand large codebases using natural language**. Upload your code, ask technical questions, and get smart answers with relevant code snippets.

---

## 🚀 Features

- 🔍 **Semantic Code Search** using LLM and vector embeddings
- 📂 Upload your Python codebase as `.zip` or folder
- 🧠 Ask natural language questions about your code (e.g., “Where is user authentication implemented?”)
- 💬 Streamlit-powered clean UI for interaction
- 📁 Embedded and stored code chunks using LangChain + FAISS

---

## 🛠️ How to Run

```
python -m venv venv
-ource venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

streamlit run app.py

```


### 1. Clone the Repository

```bash
git clone https://github.com/amarjit03/code-base-rag.git
cd code-base-rag
