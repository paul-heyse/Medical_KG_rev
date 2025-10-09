from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .artifact import StructuredArtifact


class TableCell(BaseModel):
    """Represents a cell inside a structured table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: int = Field(ge=0)
    column: int = Field(ge=0)
    content: str = Field(default="")
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    bbox: tuple[float, float, float, float] | None = Field(default=None)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Table(StructuredArtifact):
    """Structured table representation extracted from MinerU output."""

    model_config = StructuredArtifact.model_config

    cells: tuple[TableCell, ...] = Field(default_factory=tuple)
    headers: tuple[str, ...] = Field(default_factory=tuple)
    caption: str | None = None

    @field_validator("cells")
    @classmethod
    def _validate_cells(cls, value: Iterable[TableCell]) -> tuple[TableCell, ...]:
        dedup: set[tuple[int, int]] = set()
        for cell in value:
            key = (cell.row, cell.column)
            if key in dedup:
                raise ValueError(f"Duplicate table cell at row={cell.row}, column={cell.column}")
            dedup.add(key)
        return tuple(value)

    def dimensions(self) -> tuple[int, int]:
        rows = max((cell.row + cell.rowspan for cell in self.cells), default=0)
        cols = max((cell.column + cell.colspan for cell in self.cells), default=0)
        return rows, cols

    def _grid(self) -> list[list[str]]:
        rows, cols = self.dimensions()
        grid: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
        for cell in self.cells:
            for row_offset in range(cell.rowspan):
                for col_offset in range(cell.colspan):
                    grid[cell.row + row_offset][cell.column + col_offset] = cell.content
        return grid

    def to_markdown(self) -> str:
        """Render the table into GitHub-flavoured Markdown."""
        grid = self._grid()
        if not grid:
            return ""
        header = self.headers or tuple(f"Col {i+1}" for i in range(len(grid[0])))
        header_line = " | ".join(header)
        separator = " | ".join(["---"] * len(header))
        rows = [" | ".join(row) for row in grid]
        return "\n".join([header_line, separator, *rows])

    def to_html(self) -> str:
        """Render the table into a simple HTML representation."""
        grid = self._grid()
        if not grid:
            return "<table></table>"
        header = self.headers or tuple(f"Col {i+1}" for i in range(len(grid[0])))
        header_html = "".join(f"<th>{cell}</th>" for cell in header)
        body_rows = []
        for row in grid:
            body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
        return "<table><thead><tr>" + header_html + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"

    def by_row(self) -> dict[int, list[TableCell]]:
        grouped: dict[int, list[TableCell]] = defaultdict(list)
        for cell in self.cells:
            grouped[cell.row].append(cell)
        return grouped


__all__ = ["Table", "TableCell"]
