from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests


# -------------------------
# Configuration
# -------------------------

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/gemma-4-e4b"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"


# -------------------------
# Load models / database
# -------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = QdrantClient(path="qdrant_db")

print("Ready!\n")


# -------------------------
# Ask a question
# -------------------------

question = input("Ask a question: ")

print("\nSearching documents...")


# Convert question into a vector
query_vector = embedding_model.encode(
    question,
    normalize_embeddings=True,
)


# Search Qdrant
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector.tolist(),
    limit=3,
).points


# -------------------------
# Build context
# -------------------------

context_parts = []

for result in results:
    context_parts.append(
        f"Source: {result.payload['source']}\n"
        f"{result.payload['text']}"
    )

context = "\n\n---\n\n".join(context_parts)


# -------------------------
# Build prompt
# -------------------------

prompt = f"""
You are a helpful assistant answering questions about the user's documents.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I couldn't find that information in the provided documents."

Do not invent information.

Context:
{context}

Question:
{question}
"""


# -------------------------
# Ask Gemma
# -------------------------

print("Asking Gemma...\n")

response = requests.post(
    LM_STUDIO_URL,
    json={
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
    },
)

response.raise_for_status()

data = response.json()

answer = data["choices"][0]["message"]["content"]


# -------------------------
# Display answer
# -------------------------

print("=" * 60)
print("ANSWER")
print("=" * 60)
print(answer)

print("\n")
print("=" * 60)
print("SOURCES")
print("=" * 60)

for result in results:
    print(
        f"- {result.payload['source']} "
        f"(score: {result.score:.4f})"
    )