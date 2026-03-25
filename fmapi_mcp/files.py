"""File reading, MIME detection, and OpenAI content part construction.

Converts local file paths into content part dicts for the OpenAI messages API:
  {"type": "text", "text": "..."}                           — text files
  {"type": "image_url", "image_url": {"url": "data:..."}}   — image files
"""
import base64
from pathlib import Path

IMAGE_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".csv", ".xml"}
)


class FileError(Exception):
    """Raised for missing, unsupported, or invalid file inputs."""


def build_file_parts(paths: list[str]) -> list[dict]:
    """Convert file paths to OpenAI content part dicts.

    Args:
        paths: List of absolute local file paths.

    Returns:
        List of content part dicts (text or image_url).

    Raises:
        FileError: If any path is missing, a PDF, or an unsupported type.
    """
    parts: list[dict] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileError(f"File not found: {path}")
        ext = p.suffix.lower()
        if ext in IMAGE_MIME:
            mime = IMAGE_MIME[ext]
            b64 = base64.b64encode(p.read_bytes()).decode()
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        elif ext in TEXT_EXTENSIONS:
            parts.append({"type": "text", "text": p.read_text(encoding="utf-8")})
        elif ext == ".pdf":
            raise FileError("PDF files are not supported — extract text first")
        else:
            raise FileError(f"Unsupported file type: {ext}")
    return parts


def has_image_parts(paths: list[str]) -> bool:
    """Return True if any path has an image extension."""
    return any(Path(p).suffix.lower() in IMAGE_MIME for p in paths)
