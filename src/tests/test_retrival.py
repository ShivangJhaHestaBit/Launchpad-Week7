from retriever.hybrid_retriever import HybridRetriever
from pipelines.context_builder import ContextBuilder


def main():
    query = " WHAT IS REGISTRATION STATEMENT PURSUANT TO SECTION 12(b) OR (g) OF THE SECURITIES EXCHANGE ACT OF 1934"
    top_k = 5

    filters = {
        # "document_type": "policy"
        # leave empty if not needed
    }

    print("\nQuery:")
    print(query)

    print("\n Initializing retriever")
    retriever = HybridRetriever(
        index_path="src/vectorstore/index.faiss",
        metadata_path="src/data/embeddings/metadata.jsonl",
    )

    print("\n Running hybrid retrieval")
    results = retriever.search(
        query=query,
        top_k=top_k,
        filters=filters
    )

    print(f"\n Retrieved {len(results)} results\n")

    for i, r in enumerate(results, 1):
        print(f"Result {i}")
        print(f"  Score        : {r.get('rerank_score') or r.get('vector_score'):.4f}")
        print(f"  Source Type  : {r.get('source')}")
        print(f"  Metadata     : {r.get('metadata')}")
        print(f"  Text Preview : {r['text'][:300]}...")
        print("-" * 70)

    print("\n Building LLM-ready context")
    context_payload = ContextBuilder().build(results)

    print("\n Final Context (sent to LLM):\n")
    print(context_payload["context"][:1500])
    print("\n Traceable Sources:\n")

    for src in context_payload["sources"]:
        print(src)


if __name__ == "__main__":
    main()
