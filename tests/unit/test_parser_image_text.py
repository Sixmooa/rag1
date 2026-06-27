import os
from PIL import Image
from parsers.image import parse_image
from parsers.text import parse_text, parse_markdown


def test_parse_image_returns_one_image_doc(tmp_path):
    p = tmp_path / "t.png"
    Image.new("RGB", (10, 10)).save(str(p))
    docs = parse_image(str(p))
    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "image"
    assert docs[0].metadata["image_path"] == os.path.abspath(str(p))
    assert docs[0].text == ""


def test_parse_text(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("hello world\nsecond", encoding="utf-8")
    docs = parse_text(str(p))
    assert len(docs) == 1
    assert "hello world" in docs[0].text
    assert docs[0].metadata["source_type"] == "text"


def test_parse_markdown(tmp_path):
    p = tmp_path / "t.md"
    p.write_text("# Title\n\nbody", encoding="utf-8")
    docs = parse_markdown(str(p))
    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "markdown"
