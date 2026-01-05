import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import clip
import numpy as np

METADATA_FILE = Path("src/data/embeddings/image_metadata.jsonl")

OUT_DIR = Path("src/data/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EMB_FILE = OUT_DIR / "clip_image_embeddings.npy"
TEXT_EMB_FILE = OUT_DIR / "clip_text_embeddings.npy"
META_OUT_FILE = OUT_DIR / "clip_metadata.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def main():
    with open(METADATA_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    image_embeddings = []
    text_embeddings = []

    with open(META_OUT_FILE, "w", encoding="utf-8") as meta_out:
        for r in tqdm(records, desc="Generating CLIP embeddings"):
            image = preprocess(
                Image.open(r["image_path"]).convert("RGB")
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                img_emb = model.encode_image(image)
                txt_emb = model.encode_text(
                    clip.tokenize([r["caption"]]).to(device)
                )

            image_embeddings.append(img_emb.cpu().numpy()[0])
            text_embeddings.append(txt_emb.cpu().numpy()[0])

            meta_out.write(json.dumps({
                "image_path": r["image_path"],
                "ocr_text": r["ocr_text"],
                "caption": r["caption"]
            }, ensure_ascii=False) + "\n")

    image_embeddings = np.array(image_embeddings, dtype="float32")
    text_embeddings = np.array(text_embeddings, dtype="float32")

    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    np.save(IMAGE_EMB_FILE, image_embeddings)
    np.save(TEXT_EMB_FILE, text_embeddings)

    print("CLIP embeddings saved!")

if __name__ == "__main__":
    main()
