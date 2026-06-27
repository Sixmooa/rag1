import pytest
from parsers.registry import detect_type, parse, UnknownFileTypeError


def test_detect_pdf(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"")
    assert detect_type(str(p)) == "pdf"


def test_detect_image_png(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"")
    assert detect_type(str(p)) == "image"


def test_detect_image_jpg(tmp_path):
    p = tmp_path / "x.JPG"  # case-insensitive
    p.write_bytes(b"")
    assert detect_type(str(p)) == "image"


def test_detect_md(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# h\n\nhello", encoding="utf-8")
    assert detect_type(str(p)) == "markdown"


def test_detect_txt(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("plain text", encoding="utf-8")
    assert detect_type(str(p)) == "text"


def test_detect_unknown_raises(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("?", encoding="utf-8")
    with pytest.raises(UnknownFileTypeError):
        detect_type(str(p))
