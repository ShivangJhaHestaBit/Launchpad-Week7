# Architecture Overview

This document describes the end‑to‑end architecture of the **Launchpad Week 7 RAG pipeline**, covering ingestion, chunking, embedding, indexing, and retrieval.

---

## High‑Level Flow

```
PDFs
 → Ingestion
 → Chunking (token‑based)
 → Embedding (Sentence‑Transformers)
 → Vector Index (FAISS)
 → Query Engine (Similarity Search)
```

---

## Tasks Completed So Far

### 1. Document Ingestion

* PDFs loaded from `src/data/raw`
* Page‑level loading using **PyMuPDFLoader**
* Centralized ingestion logic

### 2. Chunking

* Token‑based chunking
* Chunk size: **650 tokens**
* Overlap to preserve context
* Output stored as **JSONL** (`chunks.jsonl`)

### 3. Embedding Generation

* Model: **sentence-transformers/all-MiniLM-L6-v2**
* CPU‑safe
* Output split into:

  * `embeddings.npy` → dense vectors
  * `metadata.jsonl` → text + source info

### 4. Vector Indexing

* Vector DB: **FAISS**
* Cosine similarity via normalized embeddings
* Persisted as `index.faiss`

### 5. Retrieval / Query Engine

* Query → embedding
* Top‑K similarity search on FAISS
* Metadata lookup for text reconstruction

## Project Folder Structure

```
├── src
│   ├── config
│   ├── data
│   │   ├── raw            # Original PDFs
│   │   ├── cleaned
│   │   ├── chunks         # chunks.jsonl
│   │   └── embeddings     # embeddings.npy, metadata.jsonl
│   ├── embeddings
│   │   └── embedder.py    # Embedding generation
│   ├── evaluations
│   ├── generator
│   │   ├── indexer.py     # FAISS index builder
│   │   └── llm_client.py  # LLM interface
│   ├── logs
│   ├── models
│   ├── pipelines
│   │   └── ingest.py      # Ingestion pipeline
│   ├── prompts
│   ├── retriever
│   │   └── query_engine.py # Similarity search
│   ├── utils
│   └── vectorstore
│       └── index.faiss    # FAISS vector DB
├── .gitignore
└── requirements.txt
```

**This architecture is modular, scalable, and production‑aligned.**
