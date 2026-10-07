from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("output/chunks.json")
OUTPUT_FILE = Path("output/embedded_chunks.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

# Load chunks
chunks = json.loads(
    CHUNKS_FILE.read_text(encoding="utf-8")
)

print(f"Loaded {len(chunks)} chunks")

# Get chunk text
texts = [chunk["text"] for chunk in chunks]

# Generate embeddings
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

# Add embeddings to our chunks
for chunk, embedding in zip(chunks, embeddings):
    chunk["embedding"] = embedding.tolist()

# Save
OUTPUT_FILE.write_text(
    json.dumps(
        chunks,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(f"Saved embeddings to: {OUTPUT_FILE}")
print(f"Embedding dimension: {len(embeddings[0])}")