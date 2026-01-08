import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from generator.llm_client import generate

from retriever.reranker import Reranker


class HybridRetriever:
    def __init__(
        self,
        index_path,
        metadata_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(model_name)

        with open(metadata_path, encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

        corpus_tokens = [m["text"].lower().split() for m in self.metadata]
        self.bm25 = BM25Okapi(corpus_tokens)

        self.reranker = Reranker()

    def _bm25_search(self, query, top_k):
        scores = self.bm25.get_scores(query.lower().split())

        ranked = sorted(
            zip(scores, self.metadata),
            key=lambda x: x[0],
            reverse=True
        )

        return [
            {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "bm25_score": float(score),
                "source": "bm25",
            }
            for score, doc in ranked[:top_k]
            if score > 0
        ]

    def _rrf_fusion(self, ranked_lists, top_k, k=60):
        rrf_scores = {}
        for docs in ranked_lists:
            for rank, doc in enumerate(docs):
                doc_id = (doc["text"], json.dumps(doc["metadata"], sort_keys=True))
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {
                        "doc": doc,
                        "score": 0.0,
                    }
                rrf_scores[doc_id]["score"] += 1.0 / (k + rank + 1)

        fused = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [f["doc"] for f in fused[:top_k]]

    def search(self, query, top_k=5, filters=None):
        query_emb = self.model.encode(
            query, normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            np.array([query_emb]), top_k * 3
        )

        vector_results = [
            {
                "text": self.metadata[i]["text"],
                "metadata": self.metadata[i]["metadata"],
                "vector_score": float(s),
                "source": "vector",
            }
            for s, i in zip(scores[0], indices[0])
            if i != -1
        ]   

        bm25_results = self._bm25_search(query, top_k * 2)

        fused_results = self._rrf_fusion(
            ranked_lists=[vector_results, bm25_results],
            top_k=top_k * 2
        )

        return self.reranker.rerank(query, fused_results, top_k)
