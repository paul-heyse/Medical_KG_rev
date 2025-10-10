"""Utilities for managing MinerU structured artefacts.

This module provides utilities for organizing and managing structured
artifacts extracted by the MinerU CLI, including tables, figures,
and equations. It provides efficient lookup mechanisms and metadata
aggregation for downstream processing.

Key Components:
    - ArtifactCollection: Generic container with fast ID-based lookups
    - MineruArtifacts: Aggregates all artifact types with metadata
    - build_artifacts: Factory function for creating artifact collections

Responsibilities:
    - Organize artifacts by type (tables, figures, equations)
    - Provide fast identifier-based lookups
    - Aggregate metadata for downstream processing
    - Validate artifact uniqueness by identifier
    - Support attachment resolution for references

Collaborators:
    - MinerU CLI for artifact extraction
    - Document processing pipeline for artifact consumption
    - Metadata systems for artifact storage

Side Effects:
    - Validates artifact identifier uniqueness
    - Creates immutable collections for performance
    - Aggregates metadata for external systems

Thread Safety:
    - Thread-safe: All operations are read-only after construction
    - Immutable collections prevent concurrent modification

Performance Characteristics:
    - O(1) lookup by identifier using dictionary indexing
    - O(n) construction from iterables
    - Memory efficient with tuple storage and dictionary indexing
    - Supports large artifact collections

Example:
    >>> tables = [Table(id="t1", content="..."), Table(id="t2", content="...")]
    >>> figures = [Figure(id="f1", content="...")]
    >>> artifacts = build_artifacts(tables=tables, figures=figures)
    >>> table = artifacts.tables.get("t1")
    >>> assert table is not None

"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

# ==============================================================================
# IMPORTS
# ==============================================================================
from dataclasses import dataclass
from typing import Any, Generic

from Medical_KG_rev.models.artifact import ArtifactType
from Medical_KG_rev.models.equation import Equation
from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.table import Table

# ==============================================================================
# DATA MODELS
# ==============================================================================


@dataclass(slots=True)
class ArtifactCollection(Generic[ArtifactType]):
    """Container that provides fast lookups for artefacts by identifier.

    This generic container class provides efficient storage and retrieval
    of artifacts by their unique identifiers. It maintains both a tuple
    of items for iteration and a dictionary index for O(1) lookups.

    Attributes:
        items: Immutable tuple of all artifacts in the collection
        index: Dictionary mapping artifact IDs to artifact instances

    Invariants:
        - len(items) == len(index)
        - All artifacts in items have unique IDs
        - index contains all artifacts from items

    Thread Safety:
        - Thread-safe: Immutable after construction
        - Read-only operations prevent concurrent modification

    Example:
        >>> tables = [Table(id="t1", content="..."), Table(id="t2", content="...")]
        >>> collection = ArtifactCollection.from_iterable(tables)
        >>> table = collection.get("t1")
        >>> assert table is not None
        >>> assert len(collection) == 2

    """

    items: tuple[ArtifactType, ...]
    index: dict[str, ArtifactType]

    @classmethod
    def from_iterable(cls, values: Iterable[ArtifactType]) -> ArtifactCollection[ArtifactType]:
        """Create artifact collection from iterable of artifacts.

        Args:
            values: Iterable of artifacts to include in collection

        Returns:
            New artifact collection with fast lookup capabilities

        Raises:
            ValueError: If duplicate artifact identifiers are detected

        Example:
            >>> tables = [Table(id="t1", content="..."), Table(id="t2", content="...")]
            >>> collection = ArtifactCollection.from_iterable(tables)
            >>> assert len(collection) == 2

        """
        items: list[ArtifactType] = []
        index: dict[str, ArtifactType] = {}
        for value in values:
            if value.id in index:
                raise ValueError(f"Duplicate artefact identifier detected: '{value.id}'")
            items.append(value)
            index[value.id] = value
        return cls(items=tuple(items), index=index)

    def __iter__(self) -> Iterator[ArtifactType]:
        """Iterate over all artifacts in the collection.

        Returns:
            Iterator over all artifacts in insertion order

        Example:
            >>> collection = ArtifactCollection.from_iterable(tables)
            >>> for artifact in collection:
            ...     print(artifact.id)

        """
        return iter(self.items)

    def __len__(self) -> int:
        """Get the number of artifacts in the collection.

        Returns:
            Number of artifacts in the collection

        Example:
            >>> collection = ArtifactCollection.from_iterable(tables)
            >>> assert len(collection) == 2

        """
        return len(self.items)

    def get(self, identifier: str | None) -> ArtifactType | None:
        """Get artifact by identifier.

        Args:
            identifier: Unique identifier of the artifact to retrieve

        Returns:
            Artifact instance if found, None otherwise

        Example:
            >>> collection = ArtifactCollection.from_iterable(tables)
            >>> table = collection.get("t1")
            >>> assert table is not None
            >>> missing = collection.get("nonexistent")
            >>> assert missing is None

        """
        if not identifier:
            return None
        return self.index.get(identifier)


@dataclass(slots=True)
class MineruArtifacts:
    """Aggregates structured artefacts emitted by the MinerU CLI.

    This class aggregates all structured artifacts extracted by the MinerU
    CLI, including tables, figures, and equations, along with their
    associated metadata and export information.

    Attributes:
        tables: Collection of table artifacts
        figures: Collection of figure artifacts
        equations: Collection of equation artifacts
        table_exports: Export metadata for tables
        figure_assets: Asset metadata for figures
        equation_rendering: Rendering metadata for equations

    Invariants:
        - All artifact collections contain unique identifiers
        - Metadata dictionaries are consistent with artifact collections
        - Export and asset paths reference valid artifacts

    Thread Safety:
        - Thread-safe: Immutable after construction
        - Read-only operations prevent concurrent modification

    Example:
        >>> artifacts = MineruArtifacts(
        ...     tables=ArtifactCollection.from_iterable(tables),
        ...     figures=ArtifactCollection.from_iterable(figures),
        ...     equations=ArtifactCollection.from_iterable(equations),
        ...     table_exports={},
        ...     figure_assets={},
        ...     equation_rendering={}
        ... )
        >>> table, figure, equation = artifacts.attachments_for("t1", "f1", "e1")

    """

    tables: ArtifactCollection[Table]
    figures: ArtifactCollection[Figure]
    equations: ArtifactCollection[Equation]
    table_exports: dict[str, dict[str, str]]
    figure_assets: dict[str, dict[str, str]]
    equation_rendering: dict[str, str]

    def attachments_for(
        self, table_id: str | None, figure_id: str | None, equation_id: str | None
    ) -> tuple[Table | None, Figure | None, Equation | None]:
        """Get artifacts by their identifiers.

        Args:
            table_id: Identifier of the table artifact
            figure_id: Identifier of the figure artifact
            equation_id: Identifier of the equation artifact

        Returns:
            Tuple of (table, figure, equation) artifacts, None if not found

        Example:
            >>> artifacts = MineruArtifacts(...)
            >>> table, figure, equation = artifacts.attachments_for("t1", "f1", "e1")
            >>> assert table is not None
            >>> assert figure is not None
            >>> assert equation is not None

        """
        return (
            self.tables.get(table_id),
            self.figures.get(figure_id),
            self.equations.get(equation_id),
        )

    def metadata_payload(self) -> dict[str, Any]:
        """Get metadata payload for external systems.

        Returns:
            Dictionary containing all metadata for downstream processing

        Example:
            >>> artifacts = MineruArtifacts(...)
            >>> metadata = artifacts.metadata_payload()
            >>> assert "table_exports" in metadata
            >>> assert "figure_assets" in metadata
            >>> assert "equation_rendering" in metadata

        """
        return {
            "table_exports": self.table_exports,
            "figure_assets": self.figure_assets,
            "equation_rendering": self.equation_rendering,
        }


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


def build_artifacts(
    *,
    tables: Iterable[Table],
    figures: Iterable[Figure],
    equations: Iterable[Equation],
    table_exports: dict[str, dict[str, str]] | None = None,
    figure_assets: dict[str, dict[str, str]] | None = None,
    equation_rendering: dict[str, str] | None = None,
) -> MineruArtifacts:
    """Build MinerU artifacts from individual artifact collections.

    This factory function creates a MineruArtifacts instance from
    individual iterables of artifacts and optional metadata dictionaries.
    It validates artifact uniqueness and provides default values for
    missing metadata.

    Args:
        tables: Iterable of table artifacts
        figures: Iterable of figure artifacts
        equations: Iterable of equation artifacts
        table_exports: Optional export metadata for tables
        figure_assets: Optional asset metadata for figures
        equation_rendering: Optional rendering metadata for equations

    Returns:
        MineruArtifacts instance with all artifacts and metadata

    Raises:
        ValueError: If duplicate artifact identifiers are detected

    Example:
        >>> tables = [Table(id="t1", content="..."), Table(id="t2", content="...")]
        >>> figures = [Figure(id="f1", content="...")]
        >>> equations = [Equation(id="e1", content="...")]
        >>> artifacts = build_artifacts(
        ...     tables=tables,
        ...     figures=figures,
        ...     equations=equations,
        ...     table_exports={"t1": {"csv": "path/to/t1.csv"}},
        ...     figure_assets={"f1": {"png": "path/to/f1.png"}},
        ...     equation_rendering={"e1": "LaTeX: $x^2 + y^2 = z^2$"}
        ... )
        >>> assert len(artifacts.tables) == 2
        >>> assert len(artifacts.figures) == 1
        >>> assert len(artifacts.equations) == 1

    """
    return MineruArtifacts(
        tables=ArtifactCollection.from_iterable(tables),
        figures=ArtifactCollection.from_iterable(figures),
        equations=ArtifactCollection.from_iterable(equations),
        table_exports=table_exports or {},
        figure_assets=figure_assets or {},
        equation_rendering=equation_rendering or {},
    )


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "ArtifactCollection",
    "MineruArtifacts",
    "build_artifacts",
]
