import json
import pytest
from fastapi.testclient import TestClient


@pytest.mark.slow
def test_ask_returns_sse_stream_with_retrieval_token_done():
    """Use FastAPI's TestClient with stream=True to read the SSE response."""
    from delivery.api.main import app
    client = TestClient(app)
    with client.stream("POST", "/api/ask",
                       json={"question": "GPT-3 用了多少参数", "session_id": None}) as resp:
        assert resp.status_code == 200
        events = []
        current_event = None
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line[len("event: "):]
            elif line.startswith("data: "):
                events.append((current_event, line[len("data: "):]))

    types = [t for t, _ in events]
    assert "retrieval" in types, f"missing retrieval event; got {types}"
    # Final non-ping event should be done or error (no bare 500)
    non_ping = [t for t in types if t and t != "ping"]
    assert non_ping[-1] in ("done", "error"), f"last event was {non_ping[-1]}"
