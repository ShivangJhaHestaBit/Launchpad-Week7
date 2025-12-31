# Retrieval Strategies

This system uses a hybrid retrieval approach to maximize precision and minimize hallucinations.

## 1. Vector Retrieval
- FAISS-based cosine similarity search
- Fast and scalable
- Primary recall mechanism

## 2. Keyword Fallback
- Handles:
  - Rare terms
  - IDs
  - Financial/legal jargon
- Ensures recall safety

## 3. Metadata Filters
- Enables scoped retrieval
- Example: document_type, source, domain

## 4. Deduplication
- Prevents repeated context
- Reduces token waste

## 5. Reranking
- Cross-encoder based relevance scoring
- Improves semantic precision

## 6. Traceable Context
- Every chunk includes source metadata
- Enables explainability and auditing

## Result
✔ Higher precision  
✔ Lower hallucination  
✔ Fully traceable context  
