
from pathlib import Path
import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


EMBEDDED_FILE = Path("output/embedded_chunks.json")
COLLECTION_NAME = "documents"


# Connect to local Qdrant
client = QdrantClient(path="qdrant_db")


# Load embedded chunks
chunks = json.loads(
    EMBEDDED_FILE.read_text(encoding="utf-8")
)

print(f"Loaded {len(chunks)} embedded chunks")


# Create collection if it doesn't exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection: {COLLECTION_NAME}")
else:
    print(f"Collection already exists: {COLLECTION_NAME}")


# Convert chunks into Qdrant points
points = []

for i, chunk in enumerate(chunks):
    points.append(
        PointStruct(
            id=i,  # Qdrant requires an integer or UUID
            vector=chunk["embedding"],
            payload={
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "headings": chunk["headings"],
            },
        )
    )


# Upload points
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

print(f"Uploaded {len(points)} points to Qdrant")


# Check collection
info = client.get_collection(COLLECTION_NAME)

print(f"Vectors stored: {info.points_count}")
