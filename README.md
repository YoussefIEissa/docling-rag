# 🧠 Docling RAG

> A local Retrieval-Augmented Generation (RAG) pipeline built from scratch using **Docling, Sentence Transformers, Qdrant, and Gemma 4** running locally through **LM Studio**.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Docling](https://img.shields.io/badge/Docling-Document%20Parsing-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-red?logo=qdrant)
![Gemma](https://img.shields.io/badge/Gemma%204-LLM-green)
![LM Studio](https://img.shields.io/badge/LM%20Studio-Local%20Inference-purple)
![uv](https://img.shields.io/badge/uv-Package%20Manager-blueviolet)

---

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system from the ground up.

Instead of relying solely on an LLM's pretrained knowledge, the system retrieves relevant information from a collection of documents and provides that information to the LLM as context before generating an answer.

The project was built to understand the **individual components behind modern RAG systems**, rather than hiding everything behind a high-level framework.

### 🎯 The goal

Build a complete pipeline that can:

- 📄 Process documents using **Docling**
- ✂️ Split documents into meaningful chunks
- 🧠 Convert chunks into vector embeddings
- 🗄️ Store embeddings in **Qdrant**
- 🔎 Retrieve semantically relevant information
- 🤖 Generate grounded answers using **Gemma 4**
- 💻 Run the LLM locally through **LM Studio**
- 🛡️ Avoid answering questions that cannot be supported by the provided documents

---

# 🏗️ Architecture

The complete pipeline looks like this:

```text
                    📄 Documents
                         │
                         ▼
                  ┌─────────────┐
                  │   Docling   │
                  │   Parsing   │
                  └──────┬──────┘
                         │
                         ▼
                  ✂️ HybridChunker
                         │
                         ▼
                    Text Chunks
                         │
                         ▼
              🧠 Sentence Transformer
             all-MiniLM-L6-v2
                         │
                         ▼
                 Vector Embeddings
                         │
                         ▼
                  🗄️ Qdrant
                Vector Database
                         │
                         │
                ┌────────┴────────┐
                │                 │
                │    Question     │
                │                 │
                └────────┬────────┘
                         ▼
                🧠 Query Embedding
                         │
                         ▼
                  🔎 Qdrant Search
                         │
                         ▼
                 📚 Relevant Chunks
                         │
                         ▼
                  📝 Context Prompt
                         │
                         ▼
                  🤖 Gemma 4 E4B
                         │
                         ▼
                 💬 Final Answer