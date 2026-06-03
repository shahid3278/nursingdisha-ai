import os
import chromadb
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini = genai.GenerativeModel(
    "gemini-2.5-flash"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "nursingdisha"
)

embedder = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

def ask_rag(question):

    embedding = embedder.encode(question)

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=5
    )

    context = "\n".join(
        results["documents"][0]
    )

    prompt = f"""
You are NursingDisha AI Assistant.

Use ONLY the context below.

Context:
{context}

Question:
{question}

Give a helpful answer.
"""

    response = gemini.generate_content(
        prompt
    )

    return response.text