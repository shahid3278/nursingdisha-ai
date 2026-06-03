import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "nursingdisha"
)

embedder = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

query = "best nursing colleges in bangalore with hostel"

embedding = embedder.encode(query)

results = collection.query(
    query_embeddings=[embedding.tolist()],
    n_results=5
)

print(results["documents"])