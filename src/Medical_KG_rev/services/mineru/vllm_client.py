"""HTTP client for communicating with the external vLLM server."""

from __future__ import annotations

import base64
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:  # pragma: no cover - observability logging may require optional deps
    from Medical_KG_rev.utils.logging import get_logger
except Exception:  # pragma: no cover - fallback to stdlib logging
    import logging

    class _FallbackLogger:
        def __init__(self, name: str) -> None:
            self._logger = logging.getLogger(name)

        def _format(self, message: str, details: dict[str, object]) -> str:
            if not details:
                return message
            suffix = ", ".join(f"{key}={value}" for key, value in sorted(details.items()))
            return f"{message} | {suffix}"

        def debug(self, message: str, **kwargs: object) -> None:
            self._logger.debug(self._format(message, kwargs))

        def info(self, message: str, **kwargs: object) -> None:
            self._logger.info(self._format(message, kwargs))

        def warning(self, message: str, **kwargs: object) -> None:
            self._logger.warning(self._format(message, kwargs))

        def error(self, message: str, **kwargs: object) -> None:
            self._logger.error(self._format(message, kwargs))

    def get_logger(name: str) -> _FallbackLogger:  # type: ignore[override]
        return _FallbackLogger(name)
try:  # pragma: no cover - metrics may be unavailable in lightweight envs
    from Medical_KG_rev.observability.metrics import (
        MINERU_VLLM_CLIENT_FAILURES,
        MINERU_VLLM_CLIENT_RETRIES,
        MINERU_VLLM_REQUEST_DURATION,
    )
except Exception:  # pragma: no cover - provide dummy metrics for tests without deps
    class _DummyMetric:
        def __init__(self) -> None:
            self.labels = lambda *_, **__: self  # type: ignore[assignment]

        def inc(self, *_: object, **__: object) -> None:  # type: ignore[override]
            return None

        def set(self, *_: object, **__: object) -> None:  # type: ignore[override]
            return None

        def time(self):  # type: ignore[override]
            class _Timer:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Timer()

    MINERU_VLLM_REQUEST_DURATION = _DummyMetric()
    MINERU_VLLM_CLIENT_FAILURES = _DummyMetric()
    MINERU_VLLM_CLIENT_RETRIES = _DummyMetric()
from Medical_KG_rev.services.mineru.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
)

logger = get_logger(__name__)


class VLLMClientError(Exception):
    """Base class for vLLM client errors."""


class VLLMTimeoutError(VLLMClientError):
    """Raised when the vLLM server request times out."""


class VLLMServerError(VLLMClientError):
    """Raised when the vLLM server returns an error response."""


def _record_retry(retry_state: RetryCallState) -> None:
    attempt = retry_state.attempt_number
    MINERU_VLLM_CLIENT_RETRIES.labels(retry_number=str(attempt)).inc()
    logger.warning(
        "mineru.vllm.retry",
        attempt=attempt,
        error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


class VLLMClient:
    """Async HTTP client for the vLLM OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 300.0,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
        circuit_breaker: CircuitBreaker | None = None,
        retry_attempts: int = 3,
        retry_backoff_multiplier: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_backoff_multiplier = max(0.1, float(retry_backoff_multiplier))
        logger.info(
            "mineru.vllm.client.initialised",
            base_url=self.base_url,
            timeout=timeout,
            max_connections=max_connections,
            max_keepalive=max_keepalive_connections,
            retry_attempts=self._retry_attempts,
            retry_backoff_multiplier=self._retry_backoff_multiplier,
        )

    async def __aenter__(self) -> "VLLMClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def encode_image_base64(image_bytes: bytes) -> str:
        """Return a base64 encoded string representation of image bytes."""

        return base64.b64encode(image_bytes).decode("utf-8")

    async def _post_chat_completions(self, payload: dict[str, Any]) -> httpx.Response:
        return await self.client.post("/v1/chat/completions", json=payload)

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    ) -> dict[str, Any]:
        """Execute an OpenAI-compatible chat completion request."""

        if not await self.circuit_breaker.can_execute():
            MINERU_VLLM_CLIENT_FAILURES.labels(error_type="circuit_open").inc()
            raise CircuitBreakerOpenError("Circuit breaker is open, vLLM server unavailable")

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._retry_attempts),
                wait=wait_exponential(
                    multiplier=self._retry_backoff_multiplier, min=4, max=60
                ),
                retry=retry_if_exception_type(
                    (httpx.TimeoutException, httpx.NetworkError)
                ),
                before_sleep=_record_retry,
                reraise=True,
            ):
                with attempt:
                    with MINERU_VLLM_REQUEST_DURATION.time():
                        response = await self._post_chat_completions(payload)

            response.raise_for_status()
            data = response.json()
            await self.circuit_breaker.record_success()
            logger.debug(
                "mineru.vllm.request.success",
                status_code=response.status_code,
                total_tokens=data.get("usage", {}).get("total_tokens"),
            )
            return data

        except httpx.TimeoutException as exc:
            MINERU_VLLM_CLIENT_FAILURES.labels(error_type="timeout").inc()
            await self.circuit_breaker.record_failure()
            logger.error("mineru.vllm.request.timeout", error=str(exc))
            raise VLLMTimeoutError(f"Request timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            error_type = f"http_{exc.response.status_code}"
            MINERU_VLLM_CLIENT_FAILURES.labels(error_type=error_type).inc()
            await self.circuit_breaker.record_failure()
            logger.error(
                "mineru.vllm.request.server_error",
                status_code=exc.response.status_code,
                response=exc.response.text,
            )
            raise VLLMServerError(
                f"Server error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            MINERU_VLLM_CLIENT_FAILURES.labels(error_type="network").inc()
            await self.circuit_breaker.record_failure()
            logger.error("mineru.vllm.request.network_error", error=str(exc))
            raise VLLMClientError(f"Request failed: {exc}") from exc
        except Exception as exc:
            MINERU_VLLM_CLIENT_FAILURES.labels(error_type="unknown").inc()
            await self.circuit_breaker.record_failure()
            logger.error("mineru.vllm.request.error", error=str(exc))
            raise VLLMClientError(f"Request failed: {exc}") from exc

    async def health_check(self) -> bool:
        """Return ``True`` when the vLLM server reports a healthy state."""

        try:
            response = await self.client.get("/health", timeout=5.0)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("mineru.vllm.health.failed", error=str(exc))
            return False
        return response.status_code == 200


__all__ = [
    "VLLMClient",
    "VLLMClientError",
    "VLLMServerError",
    "VLLMTimeoutError",
]
