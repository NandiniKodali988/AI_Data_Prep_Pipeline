import base64
from pathlib import Path

import anthropic
from PIL import Image

MODEL = "claude-sonnet-4-6"

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ImageProcessingAgent:
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def describe(self, image_path: Path, document_context: str = "") -> str:
        media_type = self._media_type(image_path)
        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

        context_note = f" This image appears in {document_context}." if document_context else ""
        prompt = (
            f"Describe this image in detail.{context_note} "
            "Include what it shows, any visible text or labels, the content type "
            "(diagram, chart, screenshot, etc.), and why it might be useful to someone searching this document."
        )

        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        description = resp.content[0].text.strip()
        # always use a path relative to the output dir — images/ is always next to the .md
        return f"![{image_path.name}](images/{image_path.name})\n\n> **Image description:** {description}\n"

    def _media_type(self, image_path: Path) -> str:
        ext = image_path.suffix.lower()
        if ext not in _MEDIA_TYPES:
            # pillow can handle weird formats — just convert to png
            converted = image_path.with_suffix(".png")
            Image.open(image_path).save(converted, "PNG")
            return "image/png"
        return _MEDIA_TYPES[ext]
