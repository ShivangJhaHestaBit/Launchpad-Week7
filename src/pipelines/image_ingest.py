import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pytesseract

from transformers import BlipProcessor, BlipForConditionalGeneration

RAW_DIR = Path("src/data/raw/images")
OUTPUT_FILE = Path("src/data/embeddings/image_metadata.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

def process_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")

    ocr_text = pytesseract.image_to_string(image)

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return {
        "image_path": str(image_path),
        "folder": image_path.parent.name,
        "ocr_text": ocr_text.strip(),
        "caption": caption.strip()
    }

def main():
    image_paths = [
        p for p in RAW_DIR.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"Found {len(image_paths)} images")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                record = process_image(img_path)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    print(f"Metadata saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
