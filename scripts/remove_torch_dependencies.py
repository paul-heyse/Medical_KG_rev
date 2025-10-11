#!/usr/bin/env python3
"""Script to remove torch dependencies from the main codebase.

This script removes torch imports and replaces them with gRPC service calls
or removes torch-dependent functionality entirely.
"""

import re
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Remove Torch Dependencies from Main Codebase")
console = Console()


class TorchDependencyRemover:
    """Removes torch dependencies from Python files."""

    def __init__(self, src_path: str = "src/Medical_KG_rev"):
        self.src_path = Path(src_path)
        self.torch_files: list[Path] = []
        self.modified_files: list[Path] = []
        self.errors: list[tuple[Path, str]] = []

    def find_torch_files(self) -> list[Path]:
        """Find all Python files that import torch (excluding commented lines)."""
        torch_files = []

        for py_file in self.src_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                lines = content.split("\n")

                # Check for uncommented torch imports
                has_torch_import = False
                for line in lines:
                    stripped = line.strip()
                    # Skip commented lines
                    if stripped.startswith("#"):
                        continue
                    # Check for torch imports
                    if re.match(r"^\s*(import torch|from torch)", stripped):
                        has_torch_import = True
                        break

                if has_torch_import:
                    torch_files.append(py_file)
            except Exception as e:
                self.errors.append((py_file, f"Error reading file: {e}"))

        self.torch_files = torch_files
        return torch_files

    def remove_torch_imports(self, file_path: Path) -> bool:
        """Remove torch imports from a file."""
        try:
            content = file_path.read_text()
            original_content = content

            # Remove torch imports
            lines = content.split("\n")
            new_lines = []

            for line in lines:
                # Skip lines that import torch
                if re.match(r"^\s*(import torch|from torch)", line):
                    # Add comment explaining removal
                    new_lines.append(f"# {line.strip()}  # Removed for torch isolation")
                    continue

                # Skip lines that conditionally import torch
                if "import torch" in line and ("try:" in line or "except:" in line):
                    new_lines.append(f"# {line.strip()}  # Removed for torch isolation")
                    continue

                new_lines.append(line)

            new_content = "\n".join(new_lines)

            # Only write if content changed
            if new_content != original_content:
                file_path.write_text(new_content)
                self.modified_files.append(file_path)
                return True

            return False

        except Exception as e:
            self.errors.append((file_path, f"Error removing torch imports: {e}"))
            return False

    def replace_torch_functionality(self, file_path: Path) -> bool:
        """Replace torch functionality with gRPC service calls or remove it."""
        try:
            content = file_path.read_text()
            original_content = content

            # Common torch functionality replacements
            replacements = [
                # GPU availability checks
                (
                    r"torch\.cuda\.is_available\(\)",
                    "False  # GPU functionality moved to gRPC services",
                ),
                (r"torch\.cuda\.device_count\(\)", "0  # GPU functionality moved to gRPC services"),
                (
                    r"torch\.cuda\.get_device_name\(\)",
                    '"GPU service unavailable"  # GPU functionality moved to gRPC services',
                ),
                # Tensor operations (replace with error messages)
                (
                    r"torch\.tensor\(",
                    'raise NotImplementedError("Tensor operations moved to gRPC services")  # ',
                ),
                (
                    r"torch\.zeros\(",
                    'raise NotImplementedError("Tensor operations moved to gRPC services")  # ',
                ),
                (
                    r"torch\.ones\(",
                    'raise NotImplementedError("Tensor operations moved to gRPC services")  # ',
                ),
                (
                    r"torch\.randn\(",
                    'raise NotImplementedError("Tensor operations moved to gRPC services")  # ',
                ),
                # Model operations
                (
                    r"torch\.load\(",
                    'raise NotImplementedError("Model loading moved to gRPC services")  # ',
                ),
                (
                    r"torch\.save\(",
                    'raise NotImplementedError("Model saving moved to gRPC services")  # ',
                ),
                # Device operations
                (
                    r"torch\.device\(",
                    'raise NotImplementedError("Device operations moved to gRPC services")  # ',
                ),
                (
                    r"\.to\(torch\.device",
                    'raise NotImplementedError("Device operations moved to gRPC services")  # ',
                ),
                (
                    r"\.cuda\(\)",
                    'raise NotImplementedError("CUDA operations moved to gRPC services")  # ',
                ),
                (
                    r"\.cpu\(\)",
                    'raise NotImplementedError("CPU operations moved to gRPC services")  # ',
                ),
            ]

            new_content = content
            for pattern, replacement in replacements:
                new_content = re.sub(pattern, replacement, new_content)

            # Only write if content changed
            if new_content != original_content:
                file_path.write_text(new_content)
                self.modified_files.append(file_path)
                return True

            return False

        except Exception as e:
            self.errors.append((file_path, f"Error replacing torch functionality: {e}"))
            return False

    def process_file(self, file_path: Path) -> bool:
        """Process a single file to remove torch dependencies."""
        try:
            modified = False

            # Remove torch imports
            if self.remove_torch_imports(file_path):
                modified = True

            # Replace torch functionality
            if self.replace_torch_functionality(file_path):
                modified = True

            return modified

        except Exception as e:
            self.errors.append((file_path, f"Error processing file: {e}"))
            return False

    def process_all_files(self) -> None:
        """Process all files with torch dependencies."""
        console.print("🔍 Finding files with torch dependencies...", style="blue")

        torch_files = self.find_torch_files()

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
        console.print("\n📊 Torch Dependency Removal Results", style="bold blue")

        # Summary table
        summary_table = Table(title="Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")

        summary_table.add_row("Files with torch dependencies", str(len(self.torch_files)))
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
                "\n✅ Successfully removed torch dependencies from all files.", style="green"
            )


@app.command()
def remove(
    src_path: str = typer.Option("src/Medical_KG_rev", help="Source path to process"),
    dry_run: bool = typer.Option(False, help="Show what would be changed without making changes"),
):
    """Remove torch dependencies from the main codebase."""
    if dry_run:
        console.print("🔍 Dry run mode - no changes will be made", style="yellow")

    remover = TorchDependencyRemover(src_path)

    if dry_run:
        # Just find and display files
        torch_files = remover.find_torch_files()

        if torch_files:
            console.print(
                f"📁 Found {len(torch_files)} files with torch dependencies:", style="blue"
            )

            table = Table(title="Files with Torch Dependencies")
            table.add_column("File Path", style="cyan")

            for file_path in torch_files:
                table.add_row(str(file_path))

            console.print(table)
        else:
            console.print("✅ No files with torch dependencies found", style="green")
    else:
        # Process files
        remover.process_all_files()
        remover.display_results()


@app.command()
def check(src_path: str = typer.Option("src/Medical_KG_rev", help="Source path to check")):
    """Check for remaining torch dependencies."""
    remover = TorchDependencyRemover(src_path)
    torch_files = remover.find_torch_files()

    if torch_files:
        console.print(f"❌ Found {len(torch_files)} files with torch dependencies:", style="red")

        table = Table(title="Files with Torch Dependencies")
        table.add_column("File Path", style="cyan")

        for file_path in torch_files:
            table.add_row(str(file_path))

        console.print(table)
        sys.exit(1)
    else:
        console.print("✅ No torch dependencies found in main codebase", style="green")
        sys.exit(0)


if __name__ == "__main__":
    app()
