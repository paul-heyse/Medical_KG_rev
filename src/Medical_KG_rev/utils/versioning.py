"""Utilities for handling version strings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Version:
    major: int
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"v{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, value: str) -> Version:
        if not value.startswith("v"):
            raise ValueError("Version must start with 'v'")
        parts = value[1:].split(".")
        numbers = [int(part) for part in parts]
        while len(numbers) < 3:
            numbers.append(0)
        return cls(*numbers[:3])

    def bump_minor(self) -> Version:
        return Version(self.major, self.minor + 1, 0)

    def bump_patch(self) -> Version:
        return Version(self.major, self.minor, self.patch + 1)
