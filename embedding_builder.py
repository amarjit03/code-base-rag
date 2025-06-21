import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PythonLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env
load_dotenv()

CODEBASE_DIR = "codebase"
VECTORSTORE_DIR = "vectorstore"

def load_code_documents(path=CODEBASE_DIR):
    """Load Python files as LangChain documents."""
    docs = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                print(f"[INFO] Loading: {full_path}")
                try:
                    docs.extend(PythonLoader(full_path).load())
                except Exception as e:
                    print(f"[WARNING] Failed to load {file}: {e}")
    return docs

def split_documents(docs):
    """Split large documents into smaller text chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks

def filter_valid_chunks(chunks):
    """Remove empty or non-string content chunks."""
    filtered = [
        doc for doc in chunks
        if hasattr(doc, "page_content") and isinstance(doc.page_content, str) and doc.page_content.strip()
    ]
    print(f"[INFO] {len(filtered)} valid chunks retained for embedding.")
    return filtered

def generate_embeddings(docs, persist_dir=VECTORSTORE_DIR):
    """Generate and store embeddings using Groq-compatible endpoint."""

    embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"))


    if os.path.exists(persist_dir):
        print(f"[INFO] Loading existing FAISS index from '{persist_dir}'")
        return FAISS.load_local(persist_dir, embeddings)

    # 🔍 Extract text content and filter strictly
    texts, metadatas = [], []
    for doc in docs:
        content = getattr(doc, "page_content", "")
        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned:
                texts.append(cleaned)
                metadatas.append(doc.metadata)

    if not texts:
        raise ValueError("No valid text chunks found for embedding.")

    print(f"[INFO] Final {len(texts)} valid texts being embedded...")

    # 🛠️ Validate and log sample
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            print(f"[ERROR] Chunk {i} is not a string! Type: {type(text)}")
        elif not text.strip():
            print(f"[WARNING] Chunk {i} is an empty string.")
        else:
            print(f"[DEBUG] Chunk {i} preview: {repr(text[:80])}")

    # 🚀 Perform embedding
    db = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    db.save_local(persist_dir)
    print("[SUCCESS] Embedding index created and saved.")
    return db


def build_embedding_index():
    print("[START] Building embedding index...")
    raw_docs = load_code_documents()
    chunks = split_documents(raw_docs)
    valid_chunks = filter_valid_chunks(chunks)
    if not valid_chunks:
        raise ValueError("No valid code content found for embedding.")
    generate_embeddings(valid_chunks)
    print("[DONE] Embedding pipeline completed.")

if __name__ == "__main__":
    build_embedding_index()
