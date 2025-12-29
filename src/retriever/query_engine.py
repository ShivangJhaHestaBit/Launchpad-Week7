import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

index_file = Path("src/vectorstore/index.faiss")
metadata_file = Path("src/data/embeddings/metadata.jsonl")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
top_k = 5

class QueryEngine:
    def __init__(self):
        self.index = faiss.read_index(str(index_file))

        with open(metadata_file, encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = top_k):
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        query_embedding = np.array([query_embedding], dtype="float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            result = {
                "score": float(score),
                "text": self.metadata[idx]["text"],
                "metadata": self.metadata[idx].get("metadata", {})
            }
            results.append(result)

        return results


if __name__ == "__main__":
    engine = QueryEngine()

    while True:
        query = input("\nEnter your query (or 'exit'): ")
        if query.lower() == "exit":
            break

        results = engine.search(query)

        print("\nTop Results:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. Score: {r['score']:.4f}")
            print(r["text"][:500])
            print("-" * 60)
