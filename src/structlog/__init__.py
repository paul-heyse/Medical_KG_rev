"""Lightweight structlog shim for test environments without the dependency."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

__all__ = [
    "PrintLoggerFactory",
    "configure",
    "contextvars",
    "get_logger",
    "make_filtering_bound_logger",
    "processors",
    "stdlib",
]


class _BoundLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def bind(self, **_: Any) -> _BoundLogger:
        return self

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(msg, *args, **kwargs)


class _ContextVarsModule:
    @staticmethod
    def merge_contextvars(
        _logger: Any, _method_name: str, event_dict: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return event_dict

    @staticmethod
    def bind_contextvars(**_: Any) -> None:
        return None

    @staticmethod
    def unbind_contextvars(*_: Any) -> None:
        return None


class _StdLibModule:
    @staticmethod
    def add_log_level(
        _logger: Any, method_name: str, event_dict: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if "level" not in event_dict:
            event_dict = dict(event_dict)
            event_dict["level"] = method_name
        return event_dict


class _TimeStamper:
    def __init__(self, fmt: str = "iso") -> None:
        self.fmt = fmt

    def __call__(self, _logger: Any, _method_name: str, event_dict: Mapping[str, Any]):
        event_dict = dict(event_dict)
        if "timestamp" not in event_dict:
            if self.fmt == "iso":
                event_dict["timestamp"] = datetime.utcnow().isoformat()
            else:
                event_dict["timestamp"] = datetime.utcnow().timestamp()
        return event_dict


class _JSONRenderer:
    def __init__(self, sort_keys: bool = False) -> None:
        self.sort_keys = sort_keys

    def __call__(self, _logger: Any, _method_name: str, event_dict: Mapping[str, Any]):
        return json.dumps(event_dict, sort_keys=self.sort_keys)


class _ProcessorsModule:
    TimeStamper = _TimeStamper
    JSONRenderer = _JSONRenderer


class _PrintLoggerFactory:
    def __init__(self, file: Any | None = None) -> None:
        self._file = file

    def __call__(self, name: str) -> _BoundLogger:
        return _BoundLogger(name)


def get_logger(name: str | None = None) -> _BoundLogger:
    return _BoundLogger(name or "structlog")


def configure(
    *,
    processors: list[Callable[..., Any]] | None = None,
    wrapper_class: Callable[[str], _BoundLogger] | None = None,
    logger_factory: Callable[[str], _BoundLogger] | None = None,
    cache_logger_on_first_use: bool | None = None,
) -> None:
    # The shim ignores configuration; logging is handled via the standard library.
    return None


def make_filtering_bound_logger(_level: int) -> Callable[[str], _BoundLogger]:
    def _factory(name: str) -> _BoundLogger:
        return _BoundLogger(name)

    return _factory


contextvars = _ContextVarsModule()
stdlib = _StdLibModule()
processors = _ProcessorsModule()
PrintLoggerFactory = _PrintLoggerFactory
