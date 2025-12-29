import json
import numpy as np
import faiss
from pathlib import Path

embeddings_file = Path("src/data/embeddings/embeddings.npy")
metadata_file = Path("src/data/embeddings/metadata.jsonl")
index_file = Path("src/vectorstore/index.faiss")

def main():
    embeddings = np.load(embeddings_file).astype("float32")

    with open(metadata_file, encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")
    print(f"Total vectors: {len(embeddings)}")  
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_file))


if __name__ == "__main__":
    main()
