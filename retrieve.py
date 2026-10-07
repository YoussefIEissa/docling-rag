from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Connect to local Qdrant
client = QdrantClient(path="qdrant_db")

# Load the same embedding model used for the documents
model = SentenceTransformer(MODEL_NAME)


# User's question
question = "What are the error codes on the Nova X1?"

print(f"Question: {question}")


# Convert the question into a vector
query_vector = model.encode(
    question,
    normalize_embeddings=True,
)


# Search Qdrant for the most similar chunks
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector.tolist(),
    limit=3,
).points


# Display the results
print("\nTop results:\n")

for i, result in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(f"Score: {result.score}")
    print(f"Source: {result.payload['source']}")
    print(f"Text:\n{result.payload['text']}")
    print()