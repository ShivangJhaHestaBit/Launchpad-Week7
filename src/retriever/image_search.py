import json
import faiss
from pathlib import Path
from PIL import Image
import torch
import clip

INDEX_DIR = Path("src/vectorstore")
EMBED_DIR = Path("src/data/embeddings")

IMAGE_INDEX_PATH = INDEX_DIR / "image.faiss"
TEXT_INDEX_PATH = INDEX_DIR / "text.faiss"
METADATA_PATH = EMBED_DIR / "clip_metadata.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
text_index = faiss.read_index(str(TEXT_INDEX_PATH))

with open(METADATA_PATH, encoding="utf-8") as f:
    metadata = [json.loads(line) for line in f]

def search_by_text(query: str, top_k: int = 5):
    with torch.no_grad():
        query_emb = model.encode_text(
            clip.tokenize([query]).to(device)
        ).cpu().numpy()

    faiss.normalize_L2(query_emb)
    scores, indices = image_index.search(query_emb, top_k)
    return [
        {
            **metadata[i],
            "score": float(scores[0][rank]),
            "rank": rank,
            "query_type": "text_to_image",
        }
        for rank, i in enumerate(indices[0])
        if i != -1
    ]

def search_by_image(image_path: str, top_k: int = 5):
    image = preprocess(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model.encode_image(image).cpu().numpy()

    faiss.normalize_L2(query_emb)
    scores, indices = image_index.search(query_emb, top_k)

    return [
        {
            **metadata[i],
            "score": float(scores[0][rank]),
            "rank": rank,
            "query_type": "image_to_image",
        }
        for rank, i in enumerate(indices[0])
        if i != -1
    ]

def image_to_text(image_path: str, top_k: int = 5):
    image = preprocess(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model.encode_image(image).cpu().numpy()

    faiss.normalize_L2(query_emb)
    scores, indices = text_index.search(query_emb, top_k)

    return [
        {
            "caption": metadata[i]["caption"],
            "ocr_text": metadata[i]["ocr_text"],
            "image_path": metadata[i]["image_path"],
            "score": float(scores[0][rank]),
            "rank": rank,
            "query_type": "image_to_text",
        }
        for rank, i in enumerate(indices[0])
        if i != -1
    ]