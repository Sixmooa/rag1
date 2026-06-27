import pytest
from unittest.mock import patch, MagicMock
from openai import APIStatusError
from llm.client import _call_with_retry, MAX_ATTEMPTS


def _make_status_error(status_code):
    return APIStatusError(
        message=f"err {status_code}",
        response=MagicMock(status_code=status_code, headers={}, request=MagicMock()),
        body=None,
    )


def test_retry_on_500_then_success():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _make_status_error(500)
        return "ok"

    with patch("llm.client.time.sleep"):  # skip real sleeping
        result = _call_with_retry(flaky)
    assert result == "ok"
    assert calls["n"] == 3


def test_no_retry_on_401():
    calls = {"n": 0}

    def auth_err():
        calls["n"] += 1
        raise _make_status_error(401)

    with patch("llm.client.time.sleep"):
        with pytest.raises(APIStatusError):
            _call_with_retry(auth_err)
    assert calls["n"] == 1


def test_max_attempts_constant():
    assert MAX_ATTEMPTS == 3
