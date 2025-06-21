# utils.py
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

load_dotenv()

# def create_or_load_index(persist_dir="vectorstore"):
#     embeddings = CohereEmbeddings(
#         model="embed-english-v3.0",
#         cohere_api_key=os.getenv("COHERE_API_KEY")
#     )
#     return FAISS.load_local(persist_dir, embeddings)

def create_or_load_index(persist_dir="vectorstore"):
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)


def get_llm_response(query, db):
    docs = db.similarity_search(query, k=5)
    content = "\n\n".join([doc.page_content.strip() for doc in docs if doc.page_content.strip()])

    if not content:
        return "No relevant content found."

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages = [
        {"role": "system", "content": "You are a helpful assistant for code understanding."},
        {"role": "user", "content": f"{content}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content
