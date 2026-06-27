import os
import pytest
from parsers.docling_pdf import parse_pdf

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "source")


@pytest.mark.slow
def test_parse_gpt3_paper_returns_docs_with_metadata():
    pdf_path = os.path.abspath(os.path.join(SOURCE_DIR, "GPT3_Paper.pdf"))
    if not os.path.exists(pdf_path):
        pytest.skip("GPT3_Paper.pdf not available")
    docs = parse_pdf(pdf_path)
    assert len(docs) > 0
    for d in docs:
        assert d.metadata.get("source_type") in ("pdf", "image")
        assert d.metadata.get("file_path") == pdf_path
        assert "page_number" in d.metadata
    text_docs = [d for d in docs if d.metadata.get("source_type") == "pdf"]
    assert len(text_docs) > 0
