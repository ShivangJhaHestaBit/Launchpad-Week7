class ImageContextBuilder:
    def build(self, results: list[dict]):
        if not results:
            return {
                "context": "No relevant results found.",
                "sources": []
            }

        query_type = results[0].get("query_type")

        if query_type == "text_to_image":
            return self._build_text_to_image(results)

        if query_type == "image_to_image":
            return self._build_image_to_image(results)

        if query_type == "image_to_text":
            return self._build_image_to_text(results)

        raise ValueError(f"Unknown query type: {query_type}")


    def _build_text_to_image(self, results):
        blocks = []
        sources = []

        for i, r in enumerate(results, 1):
            blocks.append(
                f"""[Image {i}]
Caption: {r.get('caption', '')}
OCR Text: {r.get('ocr_text', '')}
"""
            )

            sources.append({
                "id": i,
                "image_path": r["image_path"],
                "score": r["score"],
                "type": "text_to_image"
            })

        return {
            "context": "\n\n".join(blocks),
            "sources": sources
        }


    def _build_image_to_image(self, results):
        blocks = []
        sources = []

        for i, r in enumerate(results, 1):
            blocks.append(
                f"""[Similar Image {i}]
Caption: {r.get('caption', '')}
"""
            )

            sources.append({
                "id": i,
                "image_path": r["image_path"],
                "score": r["score"],
                "type": "image_to_image"
            })

        return {
            "context": "\n\n".join(blocks),
            "sources": sources
        }


    def _build_image_to_text(self, results):
        blocks = []
        sources = []

        for i, r in enumerate(results, 1):
            blocks.append(
                f"""[Text Result {i}]
Caption: {r.get('caption', '')}
OCR Text: {r.get('ocr_text', '')}
"""
            )

            sources.append({
                "id": i,
                "image_path": r["image_path"],
                "score": r["score"],
                "type": "image_to_text"
            })

        return {
            "context": "\n\n".join(blocks),
            "sources": sources
        }
