from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

for pdf in Path("data").glob("*.pdf"):
    print(f"Converting {pdf.name}...")
    doc = converter.convert(pdf).document
    (out_dir / f"{pdf.stem}.md").write_text(doc.export_to_markdown())

print("Done! Check the output/ folder.")