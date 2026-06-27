"""LLM 客户端工厂：DeepSeek 兼容 OpenAI 接口。
LlamaIndex OpenAILike 自带 tenacity 重试；本模块额外提供 _call_with_retry
用于手动 LLM 调用，并显式记录重试策略。"""
import logging
import time
from typing import Callable

from openai import (
    APIConnectionError, APITimeoutError, RateLimitError, APIStatusError,
)

from config.settings import settings

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_MIN = 1
BACKOFF_MAX = 8

RETRYABLE_EXC = (APIConnectionError, APITimeoutError, RateLimitError)


class _RetryableStatusError(Exception):
    """内部信号：5xx 视为可重试。"""


def _is_retryable_status(exc: APIStatusError) -> bool:
    return getattr(exc, "status_code", 0) >= 500


def _raw_call(fn: Callable):
    try:
        return fn()
    except APIStatusError as e:
        if _is_retryable_status(e):
            raise _RetryableStatusError(str(e)) from e
        raise


def _call_with_retry(fn: Callable):
    """带重试的调用：仅 5xx/超时/限流/连接错重试。4xx 直接抛。"""
    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _raw_call(fn)
        except _RetryableStatusError as e:
            last_exc = e
            logger.warning("LLM 5xx, attempt %d/%d: %s", attempt, MAX_ATTEMPTS, e)
        except RETRYABLE_EXC as e:
            last_exc = e
            logger.warning("LLM transient err, attempt %d/%d: %s", attempt, MAX_ATTEMPTS, e)
        if attempt < MAX_ATTEMPTS:
            sleep_s = min(BACKOFF_MIN * (2 ** (attempt - 1)), BACKOFF_MAX)
            time.sleep(sleep_s)
    # 重试用尽：还原原异常
    if isinstance(last_exc, _RetryableStatusError):
        cause = last_exc.__cause__
        if cause is not None:
            raise cause
        raise APIStatusError(message=str(last_exc), response=None, body=None)
    raise last_exc


_llm_singleton = None


def get_llm():
    """返回 OpenAILike LLM 单例（自带重试）。"""
    global _llm_singleton
    if _llm_singleton is None:
        from llama_index.llms.openai_like import OpenAILike
        _llm_singleton = OpenAILike(
            api_key=settings.llm.api_key,
            api_base=settings.llm.base_url,
            model=settings.llm.model,
            is_chat_model=True,
            temperature=0.3,
        )
    return _llm_singleton
