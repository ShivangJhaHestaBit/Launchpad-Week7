from retriever.image_search import (
    search_by_text,
    search_by_image,
    image_to_text,
)
from pipelines.image_context_builder import ImageContextBuilder
from generator.llm_client import generate


def run_llm(query: str, results: list, mode: str):
    context_payload = ImageContextBuilder().build(results)

    final_prompt = f"""
You are a helpful multimodal assistant.

Use ONLY the context below to answer the question.
If the answer cannot be determined, say "I don't know".

Context:
{context_payload["context"]}

Question:
{query}

Answer:
"""

    answer = generate(final_prompt)

    return {
        "mode": mode,
        "query": query,
        "answer": answer.strip(),
        "sources": context_payload["sources"],
        "context": context_payload["context"]
    }


def text_to_image_final(query: str, top_k: int = 5):
    results = search_by_text(query, top_k=top_k)
    return run_llm(query, results, "TEXT → IMAGE")


def image_to_image_final(image_path: str, top_k: int = 5):
    results = search_by_image(image_path, top_k=top_k)
    query = "Describe how these images are related"
    return run_llm(query, results, "IMAGE → IMAGE")


def image_to_text_final(image_path: str, query: str, top_k: int = 5):
    results = image_to_text(image_path, top_k=top_k)
    return run_llm(query, results, "IMAGE → TEXT")
