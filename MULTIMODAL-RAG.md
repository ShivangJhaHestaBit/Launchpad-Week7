# Multimodal Retrieval-Augmented Generation (RAG) System

This document describes the design, architecture, and workflow of the Multimodal RAG system built in this project. The system supports text and image understanding, hybrid retrieval, and fully traceable context grounding.

---

## 1. Problem Statement

Traditional RAG systems operate only on text. Real-world documents often include:

- Scanned PDFs
- Engineering diagrams
- Forms and invoices
- Images with embedded text
- Visual layouts critical to meaning

This system extends RAG to multimodal inputs by supporting:

- Text ↔ Image retrieval
- Image ↔ Image retrieval
- Image → Text answers
- OCR grounding to reduce hallucinations

---

## 2. High-Level Architecture

```
               ┌─────────────┐
               │   User      │
               │ Query       │
               └─────┬───────┘
                     │
         ┌───────────▼───────────┐
         │   Query Encoder       │
         │  (CLIP / Text Model)  │
         └───────────┬───────────┘
                     │
       ┌─────────────▼─────────────┐
       │     Vector Retrieval      │
       │  (FAISS Dense Indexes)    │
       └─────────────┬─────────────┘
                     │
       ┌─────────────▼─────────────┐
       │  Reranking & Fusion       │
       │ (Cross-Encoder / RRF)     │
       └─────────────┬─────────────┘
                     │
       ┌─────────────▼─────────────┐
       │  Grounded Context Builder │
       │(OCR + Captions + Metadata)│
       └─────────────┬─────────────┘
                     │
               ┌─────▼─────┐
               │   LLM     │
               │ Response  │
               └───────────┘
```

---

## 3. Supported Modalities

### 3.1 Ingested Data Types
- JPEG / PNG / JPG images

### 3.2 Generated Representations

Each image produces:

| Artifact               | Purpose                     |
|------------------------|-----------------------------|
| OCR Text               | Exact text grounding        |
| Caption (BLIP)         | Semantic understanding      |
| CLIP Image Embedding   | Visual semantics            |
| CLIP Text Embedding    | Cross-modal alignment       |
| Metadata               | Traceability                |

---

## 4. Ingestion Pipeline

File: `pipelines/image_ingest.py`

Responsibilities:
- Recursively scan `src/data/raw/images/**`
- For each image:
  - Run OCR (Tesseract)
  - Generate caption (BLIP)
  - Store structured metadata

Example output JSON:
```json
{
  "image_path": "...",
  "ocr_text": "...",
  "caption": "...",
  "source_doc": "...",
  "page_number": 3
}
```

---

## 5. Embedding Pipeline (CLIP)

File: `embeddings/clip_embedder.py`

Why CLIP?
- Embeds images and text into the same vector space enabling:
  - Text → Image search
  - Image → Image search
  - Image → Text retrieval

Stored embeddings and metadata:

| File                      | Content           |
|---------------------------|-------------------|
| `image_embeddings.npy`    | Image vectors     |
| `text_embeddings.npy`     | Caption vectors   |
| `clip_metadata.jsonl`     | Traceable metadata|

Separation improves performance and simplifies indexing.

---

## 6. Vector Indexing

File: `vectorstore/build_faiss.py`

Indexes created:

| Index       | Purpose                                 |
|-------------|-----------------------------------------|
| `image.faiss` | Image→Image & Text→Image retrieval    |
| `text.faiss`  | Image→Text retrieval                  |

Both use dense FAISS indexes (cosine similarity).

---

## 7. Retrieval Layer

File: `retriever/image_search.py`

Supported query modes:

1. Text → Image
   - Function: `search_by_text("voltage regulator circuit")`
   - Uses CLIP text embedding and image FAISS index

2. Image → Image
   - Function: `search_by_image("diagram.png")`
   - Uses CLIP image embedding and image FAISS index

3. Image → Text
   - Function: `image_to_text("diagram.png")`
   - Uses CLIP image embedding and text FAISS index
   - Returns captions + OCR


## 8. OCR vs Caption (Critical Distinction)

| OCR                        | Caption                    |
|----------------------------|----------------------------|
| Exact text extraction      | Semantic interpretation    |
| No understanding           | High-level meaning         |
| Keyword grounding          | Explanation                |
| Prevents hallucination     | Improves recall            |

Both are required for high-precision multimodal RAG.

---

## 9. Dense vs Sparse Retrieval

| Type      | Used Here | Purpose               |
|-----------|-----------|-----------------------|
| Dense     | Yes       | Semantic similarity   |
| Sparse    | Optional  | Keyword precision     |

Hybrid strategies (RRF, BM25 + dense) can be added later.

---

## 10. Hallucination Control

This system reduces hallucinations via:
- OCR grounding
- Caption validation
- Traceable metadata
- Source-aware retrieval
- Reranked context

Every answer can be traced back to:
image_path → caption → OCR → embedding → index

---

## 11. Folder Structure (Relevant)

```
src/
├── data/
│   ├── raw/images/
│   ├── embeddings/
│   │   ├── image_embeddings.npy
│   │   ├── text_embeddings.npy
│   │   └── clip_metadata.jsonl
├── pipelines/
│   └── image_ingest.py
├── embeddings/
│   └── clip_embedder.py
├── retriever/
│   ├── image_search.py
│   └── image_reranker.py
├── vectorstore/
│   └── build_faiss.py
```
---