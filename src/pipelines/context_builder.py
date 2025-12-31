class ContextBuilder:
    def build(self, retrieved_docs):
        context_blocks = []
        sources = []

        for i, doc in enumerate(retrieved_docs, 1):
            context_blocks.append(
                f"[Source {i}]\n{doc['text']}"
            )
            sources.append({
                "id": i,
                "metadata": doc["metadata"],
                "score": doc.get("rerank_score") or doc.get("vector_score"),
                "retrieval_type": doc.get("source")
            })

        return {
            "context": "\n\n".join(context_blocks),
            "sources": sources
        }
    