"""Utilities for parsing and manipulating semantic version strings.

Key Responsibilities:
    - Provide a lightweight ``Version`` dataclass used in release tooling

Side Effects:
    - None; operations are pure and return new instances

Thread Safety:
    - Thread-safe; dataclass instances are immutable
"""

from __future__ import annotations

from dataclasses import dataclass

# ==============================================================================
# DATA MODELS
# ==============================================================================


@dataclass(frozen=True)
class Version:
    """Semantic version representation used in tooling utilities.

    Attributes:
        major: Major version component.
        minor: Minor version component.
        patch: Patch version component.
    """

    major: int
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"v{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, value: str) -> Version:
        """Parse a string in ``vMAJOR.MINOR.PATCH`` format.

        Args:
            value: Version string beginning with ``'v'``.

        Returns:
            Parsed ``Version`` instance.

        Raises:
            ValueError: If ``value`` does not begin with ``'v'`` or contains
                non-integer components.
        """

        if not value.startswith("v"):
            raise ValueError("Version must start with 'v'")
        parts = value[1:].split(".")
        numbers = [int(part) for part in parts]
        while len(numbers) < 3:
            numbers.append(0)
        return cls(*numbers[:3])

    def bump_minor(self) -> Version:
        """Return a new version with the minor component incremented."""

        return Version(self.major, self.minor + 1, 0)

    def bump_patch(self) -> Version:
        """Return a new version with the patch component incremented."""

        return Version(self.major, self.minor, self.patch + 1)
