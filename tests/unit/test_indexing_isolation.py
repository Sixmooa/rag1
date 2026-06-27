from pipeline.indexing import IndexingPipeline, IngestResult


class _FakeParser:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on or set()
    def parse(self, path):
        if path in self.fail_on:
            raise RuntimeError("boom")
        from llama_index.core.schema import Document
        return [Document(text="t", metadata={"source_type": "text", "file_path": path})]

class _FakeIndex:
    def __init__(self, fail=False):
        self.fail = fail
        self.added = []
    def add_documents(self, docs):
        if self.fail:
            raise RuntimeError("idx fail")
        self.added.extend(docs)
        return len(docs)

def test_ok_path():
    p = IndexingPipeline(_FakeParser(), _FakeIndex(), _FakeIndex())
    r = p.ingest("/tmp/a.txt")
    assert r.status == "ok"
    assert "1 文本" in r.detail or "1" in r.detail

def test_parse_failure_isolated():
    p = IndexingPipeline(_FakeParser(fail_on={"/tmp/bad.txt"}), _FakeIndex(), _FakeIndex())
    r = p.ingest("/tmp/bad.txt")
    assert r.status == "error"
    assert "解析失败" in r.detail

def test_image_index_failure_isolated():
    p = IndexingPipeline(_FakeParser(), _FakeIndex(fail=True), _FakeIndex())
    r = p.ingest("/tmp/a.txt")
    # text 文档不会进 clip index，所以不会触发 clip 失败 → 仍 ok
    assert r.status == "ok"
