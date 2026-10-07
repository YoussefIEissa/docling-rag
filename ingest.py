from pathlib import Path
import json

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)

converter = DocumentConverter()
chunker = HybridChunker()


all_chunks = []

for pdf_path in DATA_DIR.glob("*.pdf"):
    print(f"Processing: {pdf_path.name}")

    # PDF → DoclingDocument
    doc = converter.convert(pdf_path).document

    # DoclingDocument → Hybrid chunks
    chunks = list(chunker.chunk(doc))

    print(f"  Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):

        # Metadata-enriched text
        contextualized_text = chunker.contextualize(chunk)

        all_chunks.append({
            "id": f"{pdf_path.stem}_{i}",
            "text": contextualized_text,
            "source": pdf_path.name,
            "headings": chunk.meta.headings,
        })


# Save everything
output_file = OUTPUT_DIR / "chunks.json"

output_file.write_text(
    json.dumps(
        all_chunks,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print(f"Total chunks: {len(all_chunks)}")
print(f"Saved to: {output_file}")