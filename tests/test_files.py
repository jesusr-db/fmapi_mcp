# tests/test_files.py
import base64
import os
import tempfile

import pytest

from fmapi_mcp.files import FileError, build_file_parts, has_image_parts


def make_temp(suffix: str, content: bytes | str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(content if isinstance(content, bytes) else content.encode())
    f.close()
    return f.name


# --- build_file_parts ---

def test_text_file_returns_text_part():
    path = make_temp(".txt", "hello world")
    try:
        parts = build_file_parts([path])
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert parts[0]["text"] == "hello world"
    finally:
        os.unlink(path)


def test_md_file_returns_text_part():
    path = make_temp(".md", "# heading")
    try:
        parts = build_file_parts([path])
        assert parts[0]["type"] == "text"
    finally:
        os.unlink(path)


def test_png_returns_image_url_part():
    path = make_temp(".png", b"\x89PNG\r\n\x1a\n")
    try:
        parts = build_file_parts([path])
        assert len(parts) == 1
        assert parts[0]["type"] == "image_url"
        url = parts[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # Verify base64 payload decodes to the original bytes
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == b"\x89PNG\r\n\x1a\n"
    finally:
        os.unlink(path)


def test_jpg_uses_jpeg_mime():
    path = make_temp(".jpg", b"\xff\xd8\xff")
    try:
        parts = build_file_parts([path])
        assert "image/jpeg" in parts[0]["image_url"]["url"]
    finally:
        os.unlink(path)


def test_jpeg_uses_jpeg_mime():
    path = make_temp(".jpeg", b"\xff\xd8\xff")
    try:
        parts = build_file_parts([path])
        assert "image/jpeg" in parts[0]["image_url"]["url"]
    finally:
        os.unlink(path)


def test_gif_uses_gif_mime():
    path = make_temp(".gif", b"GIF89a")
    try:
        parts = build_file_parts([path])
        assert "image/gif" in parts[0]["image_url"]["url"]
    finally:
        os.unlink(path)


def test_webp_uses_webp_mime():
    path = make_temp(".webp", b"RIFF")
    try:
        parts = build_file_parts([path])
        assert "image/webp" in parts[0]["image_url"]["url"]
    finally:
        os.unlink(path)


def test_pdf_raises_file_error():
    path = make_temp(".pdf", b"%PDF-1.4")
    try:
        with pytest.raises(FileError, match="PDF files are not supported"):
            build_file_parts([path])
    finally:
        os.unlink(path)


def test_unsupported_type_raises_file_error():
    path = make_temp(".docx", b"content")
    try:
        with pytest.raises(FileError, match=r"Unsupported file type: \.docx"):
            build_file_parts([path])
    finally:
        os.unlink(path)


def test_binary_file_with_text_extension_raises_file_error():
    path = make_temp(".txt", b"\xff\xfe binary content \x00\x01")
    try:
        with pytest.raises(FileError, match="not valid UTF-8"):
            build_file_parts([path])
    finally:
        os.unlink(path)


def test_missing_file_raises_file_error():
    with pytest.raises(FileError, match="File not found"):
        build_file_parts(["/tmp/does_not_exist_fmapi_xyz_123.txt"])


def test_multiple_files_returns_multiple_parts():
    p1 = make_temp(".txt", "file one")
    p2 = make_temp(".md", "file two")
    try:
        parts = build_file_parts([p1, p2])
        assert len(parts) == 2
        assert all(p["type"] == "text" for p in parts)
    finally:
        os.unlink(p1)
        os.unlink(p2)


def test_empty_list_returns_empty():
    assert build_file_parts([]) == []


# --- has_image_parts ---

def test_has_image_parts_true_for_image():
    assert has_image_parts(["photo.png"]) is True
    assert has_image_parts(["img.jpg"]) is True


def test_has_image_parts_false_for_text():
    assert has_image_parts(["readme.txt"]) is False
    assert has_image_parts([]) is False


def test_has_image_parts_true_if_any_is_image():
    assert has_image_parts(["readme.txt", "photo.webp"]) is True
