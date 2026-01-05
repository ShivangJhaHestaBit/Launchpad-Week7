import faiss
import numpy as np
from pathlib import Path

IMAGE_EMBEDDINGS = Path("src/data/embeddings/clip_image_embeddings.npy")
TEXT_EMBEDDINGS = Path("src/data/embeddings/clip_text_embeddings.npy")

IMAGE_INDEX_OUT = Path("src/vectorstore/image.faiss")
TEXT_INDEX_OUT = Path("src/vectorstore/text.faiss")

IMAGE_INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)


def build_faiss_index(embedding_path: Path, index_out: Path):
    
    embeddings = np.load(embedding_path).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(embeddings)

    index.add(embeddings)
    faiss.write_index(index, str(index_out))

    print(f"Saved FAISS index to: {index_out}")
    print(f"Vectors indexed: {index.ntotal}")


def main():
    print("Building image FAISS index")
    build_faiss_index(IMAGE_EMBEDDINGS, IMAGE_INDEX_OUT)

    print("\nBuilding text FAISS index")
    build_faiss_index(TEXT_EMBEDDINGS, TEXT_INDEX_OUT)


if __name__ == "__main__":
    main()
