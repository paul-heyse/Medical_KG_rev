"""Dependency providers for presentation layer components."""

from __future__ import annotations

from functools import lru_cache

from .interface import ResponsePresenter
from .jsonapi import JSONAPIPresenter


@lru_cache(maxsize=1)
def _default_presenter() -> ResponsePresenter:
    return JSONAPIPresenter()


def get_response_presenter() -> ResponsePresenter:
    """Return the default response presenter instance."""

    return _default_presenter()
