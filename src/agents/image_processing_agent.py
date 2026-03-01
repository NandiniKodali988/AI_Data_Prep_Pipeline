"""Image Processing Agent — describes images using Claude Vision."""
import base64
from pathlib import Path

import anthropic
from PIL import Image


class ImageProcessingAgent:
    """
    Sends images to Claude Vision and returns a Markdown description.

    Images are encoded as base64 and passed to claude-sonnet-4-6 with
    document context so descriptions are relevant and searchable.
    """

    MODEL = "claude-sonnet-4-6"

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def describe(self, image_path: Path, document_context: str = "") -> str:
        """
        Generate a Markdown description for an image file.

        Args:
            image_path: Path to the image file (PNG, JPEG, etc.)
            document_context: Context string, e.g. document title and type.

        Returns:
            Markdown string with the image description wrapped in a blockquote.
        """
        media_type = self._get_media_type(image_path)
        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

        context_clause = (
            f" This image appears in {document_context}." if document_context else ""
        )

        prompt = (
            f"Describe this image in detail.{context_clause} "
            "Include: what the image shows, any visible text or labels, "
            "the type of content (diagram, chart, screenshot, photo, etc.), "
            "and why it might be relevant to someone searching this document."
        )

        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        description = response.content[0].text.strip()
        return f"![{image_path.name}]({image_path})\n\n> **Image description:** {description}\n"

    def _get_media_type(self, image_path: Path) -> str:
        ext = image_path.suffix.lower()
        mapping = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        if ext not in mapping:
            # Convert to PNG via Pillow for unsupported types
            converted = image_path.with_suffix(".png")
            Image.open(image_path).save(converted, "PNG")
            return "image/png"
        return mapping[ext]
