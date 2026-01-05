from retriever.image_search import (
    search_by_text,
    search_by_image,
    image_to_text,
)

TEST_IMAGE = "sample_1.jpeg"
TOP_K = 5


def print_results(title, results):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    for r in results:
        print(f"Rank: {r['rank']} | Score: {r['score']:.4f}")
        print(f"Image: {r.get('image_path')}")
        print(f"Caption: {r.get('caption')}")
        print("-" * 60)


def text_to_image():
    print("\nRunning TEXT → IMAGE search")
    query = "engineering diagram of a hydraulic system"
    results = search_by_text(query, top_k=TOP_K)
    print_results("TEXT → IMAGE RESULTS", results)


def image_to_image():
    print("\nRunning IMAGE → IMAGE search")
    results = search_by_image(TEST_IMAGE, top_k=TOP_K)
    print_results("IMAGE → IMAGE RESULTS", results)


def test_image_to_text():
    print("\nRunning IMAGE → TEXT search")
    results = image_to_text(TEST_IMAGE, top_k=TOP_K)

    print(f"\n{'=' * 60}")
    print("IMAGE → TEXT RESULTS")
    print(f"{'=' * 60}")

    for r in results:
        print(f"Rank: {r['rank']} | Score: {r['score']:.4f}")
        print(f"Caption: {r['caption']}")
        print(f"OCR Text (preview): {r['ocr_text'][:200]}...")
        print("-" * 60)


def main():
    text_to_image()
    image_to_image()
    test_image_to_text()


if __name__ == "__main__":
    main()
