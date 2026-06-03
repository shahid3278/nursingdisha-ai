import requests
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="nursingdisha"
)

embedder = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

url = "https://nursingdisha.com/wp-json/nursing-ai/v1/all-colleges"

data = requests.get(url).json()

for college in data:

    doc = f"""
    College Name: {college['name']}
    City: {college['city']}
    Rating: {college['rating']}
    Hostel: {college['hostel']}
    Description: {college['description']}
    """

    embedding = embedder.encode(doc)

    collection.add(
        ids=[str(college["id"])],
        documents=[doc],
        embeddings=[embedding.tolist()]
    )

print("Data inserted successfully")