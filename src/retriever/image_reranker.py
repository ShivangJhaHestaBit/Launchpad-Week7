from sentence_transformers import CrossEncoder

class ImageReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query_text: str, candidates: list, top_k: int = 5):
        pairs = []

        for c in candidates:
            doc_text = f"{c.get('caption', '')} {c.get('ocr_text', '')}"
            pairs.append((query_text, doc_text))

        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]
