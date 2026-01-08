from retriever.hybrid_retriever import HybridRetriever
from pipelines.context_builder import ContextBuilder
from generator.llm_client import generate

def ask_retrival(query: str):
    top_k = 5
    filters = {
        # "document_type": "policy"
        # leave empty if not needed
    }

    # print("\nQuery:")
    # print(query)

    # print("\n Initializing retriever")
    retriever = HybridRetriever(
        index_path="src/vectorstore/index.faiss",
        metadata_path="src/data/embeddings/metadata.jsonl",
    )

    # print("\n Running hybrid retrieval")
    results = retriever.search(
        query=query,
        top_k=top_k,
        filters=filters
    )

    # print(f"\n Retrieved {len(results)} results\n")

    # for i, r in enumerate(results, 1):
    #     print(f"Result {i}")
    #     print(f"  Score        : {r.get('rerank_score') or r.get('vector_score'):.4f}")
    #     print(f"  Source Type  : {r.get('source')}")
    #     print(f"  Metadata     : {r.get('metadata')}")
    #     print(f"  Text Preview : {r['text'][:300]}...")
    #     print("-" * 70)

    # print("\n Building LLM-ready context")
    context_payload = ContextBuilder().build(results)

    # print("\n Final Context (sent to LLM):\n")
    # print(context_payload["context"][:1500])
    final_prompt = f"""
You are a helpful and factual assistant.

Use ONLY the context below to answer the question.
If the answer is not present, say "I don't know".

Context:
{context_payload['context'][:1000]}

Question:
{query}

Answer:
"""
    answer = generate(final_prompt)
    # print("===============================================================================")
    # print(answer)
    return {
        "query": query,
        "answer": answer.strip(),
        "context": context_payload['context'][:1000]
    }
