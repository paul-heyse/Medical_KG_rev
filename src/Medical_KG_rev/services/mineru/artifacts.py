"""Utilities for managing MinerU structured artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, Iterator

from Medical_KG_rev.models.artifact import ArtifactType
from Medical_KG_rev.models.equation import Equation
from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.table import Table


@dataclass(slots=True)
class ArtifactCollection(Generic[ArtifactType]):
    """Container that provides fast lookups for artefacts by identifier."""

    items: tuple[ArtifactType, ...]
    index: Dict[str, ArtifactType]

    @classmethod
    def from_iterable(cls, values: Iterable[ArtifactType]) -> ArtifactCollection[ArtifactType]:
        items: list[ArtifactType] = []
        index: Dict[str, ArtifactType] = {}
        for value in values:
            if value.id in index:
                raise ValueError(f"Duplicate artefact identifier detected: '{value.id}'")
            items.append(value)
            index[value.id] = value
        return cls(items=tuple(items), index=index)

    def __iter__(self) -> Iterator[ArtifactType]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def get(self, identifier: str | None) -> ArtifactType | None:
        if not identifier:
            return None
        return self.index.get(identifier)


@dataclass(slots=True)
class MineruArtifacts:
    """Aggregates structured artefacts emitted by the MinerU CLI."""

    tables: ArtifactCollection[Table]
    figures: ArtifactCollection[Figure]
    equations: ArtifactCollection[Equation]
    table_exports: dict[str, dict[str, str]]
    figure_assets: dict[str, dict[str, str]]
    equation_rendering: dict[str, str]

    def attachments_for(
        self, table_id: str | None, figure_id: str | None, equation_id: str | None
    ) -> tuple[Table | None, Figure | None, Equation | None]:
        return (
            self.tables.get(table_id),
            self.figures.get(figure_id),
            self.equations.get(equation_id),
        )

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "table_exports": self.table_exports,
            "figure_assets": self.figure_assets,
            "equation_rendering": self.equation_rendering,
        }


def build_artifacts(
    *,
    tables: Iterable[Table],
    figures: Iterable[Figure],
    equations: Iterable[Equation],
    table_exports: dict[str, dict[str, str]] | None = None,
    figure_assets: dict[str, dict[str, str]] | None = None,
    equation_rendering: dict[str, str] | None = None,
) -> MineruArtifacts:
    return MineruArtifacts(
        tables=ArtifactCollection.from_iterable(tables),
        figures=ArtifactCollection.from_iterable(figures),
        equations=ArtifactCollection.from_iterable(equations),
        table_exports=table_exports or {},
        figure_assets=figure_assets or {},
        equation_rendering=equation_rendering or {},
    )


__all__ = [
    "ArtifactCollection",
    "MineruArtifacts",
    "build_artifacts",
]
