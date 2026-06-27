import json
from pipeline.generation import SSEEvent


def test_token_event_serializes_to_sse_frame():
    e = SSEEvent(type="token", data="你")
    frame = e.to_sse()
    assert frame == 'event: token\ndata: "你"\n\n'

def test_retrieval_event_serializes_dict_data():
    e = SSEEvent(type="retrieval", data=[{"file": "a.pdf", "page": 1}])
    frame = e.to_sse()
    assert frame.startswith("event: retrieval\n")
    assert "a.pdf" in frame
    assert frame.endswith("\n\n")

def test_error_event():
    e = SSEEvent(type="error", data={"message": "boom", "type": "RuntimeError"})
    frame = e.to_sse()
    assert "event: error" in frame
    parsed = json.loads(frame.split("data: ", 1)[1].strip())
    assert parsed["message"] == "boom"
