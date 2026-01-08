import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class RAGEvaluator:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def faithfulness_score(self, answer: str, context: str) -> float:
        answer_emb = self.model.encode(answer)
        context_emb = self.model.encode(context)

        score = self._cosine(answer_emb, context_emb)

        return round(float(score), 3)
