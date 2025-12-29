import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

chunks_file = Path("src/data/chunks/chunks.jsonl")
output_file = Path("src/data/embeddings/embeddings.npy")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
metadata_file = Path("src/data/embeddings/metadata.jsonl")

batch_size = 32

def load_chunks(path: Path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    chunks = load_chunks(chunks_file)
    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    np.save(output_file, embeddings)
    with open(metadata_file, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks):
            record = {
                "id": chunk.get("id"),
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {})
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()