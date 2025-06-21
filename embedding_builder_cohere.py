# embedding_builder.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PythonLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

load_dotenv()

CODEBASE_PATH = "codebase"
VECTORSTORE_PATH = "vectorstore"

def load_code_documents(path=CODEBASE_PATH):
    docs = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                docs.extend(PythonLoader(full_path).load())
    return docs

def build_embedding_index():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    raw_docs = load_code_documents()
    chunks = splitter.split_documents(raw_docs)
    valid_chunks = [doc for doc in chunks if doc.page_content.strip()]

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    db = FAISS.from_texts(
        texts=[doc.page_content.strip() for doc in valid_chunks],
        embedding=embeddings,
        metadatas=[doc.metadata for doc in valid_chunks]
    )
    db.save_local(VECTORSTORE_PATH)
    print("[SUCCESS] Embeddings created and saved.")

if __name__ == "__main__":
    build_embedding_index()
