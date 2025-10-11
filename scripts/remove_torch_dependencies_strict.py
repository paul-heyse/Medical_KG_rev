#!/usr/bin/env python3
"""Strict script to remove ALL torch dependencies from the codebase.

This script removes ALL torch-dependent functionality and replaces it with
appropriate gRPC service calls or removes it entirely.
"""

import re
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Remove ALL Torch Dependencies - Strict Approach")
console = Console()


class StrictTorchRemover:
    """Removes ALL torch dependencies with strict replacements."""

    def __init__(self):
        self.modified_files: list[Path] = []
        self.errors: list[tuple[Path, str]] = []

    def process_file(self, file_path: Path) -> bool:
        """Process a single file to remove ALL torch dependencies."""
        try:
            content = file_path.read_text()
            original_content = content

            # Remove ALL torch imports
            lines = content.split("\n")
            new_lines = []

            for line in lines:
                # Remove torch imports
                if re.match(r"^\s*(import torch|from torch)", line):
                    new_lines.append(f"# {line.strip()}  # Removed for torch isolation")
                    continue

                # Remove conditional torch imports
                if "import torch" in line and ("try:" in line or "except:" in line):
                    new_lines.append(f"# {line.strip()}  # Removed for torch isolation")
                    continue

                # Replace ALL torch functionality with NotImplementedError
                if "torch." in line:
                    # Replace torch calls with NotImplementedError
                    new_line = re.sub(
                        r"torch\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)",
                        'raise NotImplementedError("Torch functionality moved to gRPC services")',
                        line,
                    )
                    if new_line != line:
                        new_lines.append(f"# {line.strip()}  # Replaced for torch isolation")
                        new_lines.append(new_line)
                        continue

                # Replace torch variable assignments
                if re.match(r"^\s*torch\s*=", line):
                    new_lines.append(f"# {line.strip()}  # Removed for torch isolation")
                    new_lines.append("torch = None  # Torch functionality moved to gRPC services")
                    continue

                # Replace torch attribute access
                if re.search(r"\btorch\.[a-zA-Z_][a-zA-Z0-9_]*", line):
                    new_line = re.sub(
                        r"\btorch\.[a-zA-Z_][a-zA-Z0-9_]*",
                        'raise NotImplementedError("Torch functionality moved to gRPC services")',
                        line,
                    )
                    if new_line != line:
                        new_lines.append(f"# {line.strip()}  # Replaced for torch isolation")
                        new_lines.append(new_line)
                        continue

                new_lines.append(line)

            new_content = "\n".join(new_lines)

            if new_content != original_content:
                file_path.write_text(new_content)
                self.modified_files.append(file_path)
                return True

            return False

        except Exception as e:
            self.errors.append((file_path, f"Error processing file: {e}"))
            return False

    def process_all_files(self, src_path: str = "src/Medical_KG_rev") -> None:
        """Process all files with torch dependencies."""
        src_path = Path(src_path)

        # Find all Python files with torch imports
        torch_files = []
        for py_file in src_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                if "import torch" in content or "from torch" in content or "torch." in content:
                    torch_files.append(py_file)
            except Exception:
                continue

        if not torch_files:
            console.print("✅ No files with torch dependencies found", style="green")
            return

        console.print(f"📁 Found {len(torch_files)} files with torch dependencies", style="blue")

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:

            task = progress.add_task("Processing files...", total=len(torch_files))

            for file_path in torch_files:
                progress.update(task, description=f"Processing {file_path.name}...")

                try:
                    modified = self.process_file(file_path)
                    if modified:
                        console.print(f"✅ Modified: {file_path}", style="green")
                    else:
                        console.print(f"ℹ️  No changes needed: {file_path}", style="blue")

                except Exception as e:
                    console.print(f"❌ Error processing {file_path}: {e}", style="red")
                    self.errors.append((file_path, str(e)))

                progress.advance(task)

    def display_results(self) -> None:
        """Display the results of the torch removal process."""
        console.print("\n📊 Strict Torch Dependency Removal Results", style="bold blue")

        # Summary table
        summary_table = Table(title="Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")

        summary_table.add_row("Files modified", str(len(self.modified_files)))
        summary_table.add_row("Errors encountered", str(len(self.errors)))

        console.print(summary_table)

        # Modified files table
        if self.modified_files:
            modified_table = Table(title="Modified Files")
            modified_table.add_column("File Path", style="cyan")
            modified_table.add_column("Status", style="green")

            for file_path in self.modified_files:
                modified_table.add_row(str(file_path), "✅ Modified")

            console.print(modified_table)

        # Errors table
        if self.errors:
            error_table = Table(title="Errors")
            error_table.add_column("File Path", style="cyan")
            error_table.add_column("Error", style="red")

            for file_path, error in self.errors:
                error_table.add_row(str(file_path), error)

            console.print(error_table)

        # Overall result
        if self.errors:
            console.print(
                f"\n⚠️  Completed with {len(self.errors)} errors. Please review the errors above.",
                style="yellow",
            )
        else:
            console.print(
                "\n✅ Successfully removed ALL torch dependencies from all files.", style="green"
            )


@app.command()
def remove(src_path: str = typer.Option("src/Medical_KG_rev", help="Source path to process")):
    """Remove ALL torch dependencies with strict replacements."""
    remover = StrictTorchRemover()
    remover.process_all_files(src_path)
    remover.display_results()


if __name__ == "__main__":
    app()
