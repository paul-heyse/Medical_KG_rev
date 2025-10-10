#!/usr/bin/env python3
"""Test real PDF processing with MinerU using downloaded papers.

This script demonstrates the complete workflow:
1. Use downloaded PDFs from previous runs
2. Process them with MinerU (real or simulated)
3. Generate Markdown and JSON outputs
4. Show the full processing pipeline

Usage:
    python test_real_pdf_processing.py

Environment Variables:
    PYALEX_EMAIL: Email address for pyalex API access (default: paul@heyse.io)
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest


class RealPDFProcessor:
    """Process real PDFs with MinerU."""

    def __init__(self, input_dir: str = "random_papers_output/pdfs", output_dir: str = "real_pdf_output"):
        """Initialize the processor."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.markdown_dir = self.output_dir / "markdown"
        self.json_dir = self.output_dir / "json"
        self.reports_dir = self.output_dir / "reports"

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.markdown_dir.mkdir(exist_ok=True)
        self.json_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        # Initialize MinerU processor
        self.mineru_processor = MineruProcessor()

        # Results tracking
        self.results = {
            "pdfs_found": 0,
            "pdfs_processed": 0,
            "processing_errors": 0,
            "processing_details": []
        }

    def find_pdfs(self) -> List[Path]:
        """Find PDF files in the input directory."""
        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            return []

        pdfs = list(self.input_dir.glob("*.pdf"))
        self.results["pdfs_found"] = len(pdfs)

        print(f"📁 Found {len(pdfs)} PDF files in {self.input_dir}")
        for pdf in pdfs:
            print(f"   {pdf.name} ({pdf.stat().st_size} bytes)")

        return pdfs

    def process_pdf_with_mineru(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a PDF using MinerU."""
        try:
            print(f"🔄 Processing PDF: {pdf_path.name}")

            # Read PDF content
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()

            # Create clean document ID
            doc_id = pdf_path.stem

            # Create MinerU request
            request = MineruRequest(
                tenant_id="real-pdf-test",
                document_id=doc_id,
                content=pdf_content
            )

            # Process with MinerU
            start_time = time.time()
            response = self.mineru_processor.process(request)
            processing_time = time.time() - start_time

            # Generate outputs
            markdown_content = self.generate_markdown(response, pdf_path)
            json_content = self.generate_json(response, pdf_path)

            # Save outputs
            markdown_file = self.markdown_dir / f"{doc_id}.md"
            json_file = self.json_dir / f"{doc_id}.json"

            with open(markdown_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            with open(json_file, "w", encoding="utf-8") as f:
                f.write(json_content)

            # Create processing result
            result = {
                "pdf_file": str(pdf_path),
                "document_id": response.document.document_id if response.document else None,
                "processing_time_seconds": processing_time,
                "blocks_count": len(response.document.blocks) if response.document else 0,
                "tables_count": len(response.document.tables) if response.document else 0,
                "figures_count": len(response.document.figures) if response.document else 0,
                "equations_count": len(response.document.equations) if response.document else 0,
                "markdown_file": str(markdown_file),
                "json_file": str(json_file),
                "worker_id": response.metadata.worker_id if response.metadata else None,
                "success": True
            }

            print(f"✅ Processed: {pdf_path.name} ({processing_time:.2f}s)")
            print(f"   Blocks: {result['blocks_count']}")
            print(f"   Tables: {result['tables_count']}")
            print(f"   Figures: {result['figures_count']}")
            print(f"   Markdown: {markdown_file.name}")
            print(f"   JSON: {json_file.name}")

            return result

        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            return {
                "pdf_file": str(pdf_path),
                "error": str(e),
                "success": False
            }

    def generate_markdown(self, response, pdf_path: Path) -> str:
        """Generate Markdown content from MinerU response."""
        if not response.document:
            return f"# Error processing {pdf_path.name}\n\nProcessing failed."

        doc = response.document
        doc_id = pdf_path.stem

        markdown = f"# {doc_id}\n\n"
        markdown += f"**Document ID:** {doc.document_id}\n"
        markdown += f"**Tenant ID:** {doc.tenant_id}\n"
        markdown += f"**Processing Time:** {response.metadata.processing_time_ms if response.metadata else 'Unknown'} ms\n\n"

        # Add blocks
        if doc.blocks:
            markdown += "## Document Blocks\n\n"
            for i, block in enumerate(doc.blocks):
                if block.text:
                    markdown += f"### Block {i+1} (Page {block.page})\n"
                    markdown += f"**Type:** {block.kind}\n"
                    if block.confidence:
                        markdown += f"**Confidence:** {block.confidence:.2f}\n"
                    markdown += f"\n{block.text}\n\n"

        # Add tables
        if doc.tables:
            markdown += "## Tables\n\n"
            for i, table in enumerate(doc.tables):
                markdown += f"### Table {i+1}\n"
                if table.title:
                    markdown += f"**Title:** {table.title}\n"
                markdown += f"**Rows:** {table.row_count}\n"
                markdown += f"**Columns:** {table.column_count}\n\n"

        # Add figures
        if doc.figures:
            markdown += "## Figures\n\n"
            for i, figure in enumerate(doc.figures):
                markdown += f"### Figure {i+1}\n"
                if figure.title:
                    markdown += f"**Title:** {figure.title}\n"
                markdown += f"**Page:** {figure.page}\n\n"

        # Add equations
        if doc.equations:
            markdown += "## Equations\n\n"
            for i, equation in enumerate(doc.equations):
                markdown += f"### Equation {i+1}\n"
                if equation.latex:
                    markdown += f"**LaTeX:** `{equation.latex}`\n"
                markdown += f"**Page:** {equation.page}\n\n"

        return markdown

    def generate_json(self, response, pdf_path: Path) -> str:
        """Generate JSON content from MinerU response."""
        import json

        if not response.document:
            return json.dumps({
                "error": f"Processing failed for {pdf_path.name}",
                "success": False
            }, indent=2)

        doc = response.document

        # Convert document to serializable format
        json_data = {
            "document_id": doc.document_id,
            "tenant_id": doc.tenant_id,
            "blocks": [
                {
                    "id": block.id,
                    "page": block.page,
                    "kind": block.kind,
                    "text": block.text,
                    "bbox": block.bbox,
                    "confidence": block.confidence,
                    "reading_order": block.reading_order,
                    "metadata": block.metadata
                }
                for block in doc.blocks
            ],
            "tables": [
                {
                    "id": table.id,
                    "page": table.page,
                    "title": table.title,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "cells": [
                        {
                            "row": cell.row,
                            "column": cell.column,
                            "content": cell.content,
                            "rowspan": cell.rowspan,
                            "colspan": cell.colspan
                        }
                        for cell in table.cells
                    ]
                }
                for table in doc.tables
            ],
            "figures": [
                {
                    "id": figure.id,
                    "page": figure.page,
                    "title": figure.title,
                    "bbox": figure.bbox,
                    "metadata": figure.metadata
                }
                for figure in doc.figures
            ],
            "equations": [
                {
                    "id": equation.id,
                    "page": equation.page,
                    "latex": equation.latex,
                    "bbox": equation.bbox,
                    "metadata": equation.metadata
                }
                for equation in doc.equations
            ],
            "metadata": doc.metadata,
            "provenance": doc.provenance,
            "processing_metadata": {
                "worker_id": response.metadata.worker_id if response.metadata else None,
                "processing_time_ms": response.metadata.processing_time_ms if response.metadata else None,
                "gpu_utilization": response.metadata.gpu_utilization if response.metadata else None,
                "mineru_version": response.metadata.mineru_version if response.metadata else None,
                "model_names": response.metadata.model_names if response.metadata else {},
                "gpu_id": response.metadata.gpu_id if response.metadata else None,
                "started_at": response.metadata.started_at.isoformat() if response.metadata and response.metadata.started_at else None,
                "completed_at": response.metadata.completed_at.isoformat() if response.metadata and response.metadata.completed_at else None,
                "duration_seconds": response.metadata.duration_seconds if response.metadata else None,
            },
            "success": True
        }

        return json.dumps(json_data, indent=2, default=str)

    def process_all_pdfs(self) -> None:
        """Process all PDFs in the input directory."""
        print("="*70)
        print("REAL PDF PROCESSING WITH MINERU")
        print("="*70)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("-"*70)

        # Find PDFs
        pdfs = self.find_pdfs()
        if not pdfs:
            print("❌ No PDFs to process")
            return

        # Process each PDF
        print(f"\n🔄 Processing {len(pdfs)} PDFs...")
        for pdf_path in pdfs:
            try:
                result = self.process_pdf_with_mineru(pdf_path)
                self.results["processing_details"].append(result)

                if result["success"]:
                    self.results["pdfs_processed"] += 1
                else:
                    self.results["processing_errors"] += 1

            except Exception as e:
                print(f"❌ Unexpected error processing {pdf_path.name}: {e}")
                self.results["processing_errors"] += 1
                self.results["processing_details"].append({
                    "pdf_file": str(pdf_path),
                    "error": str(e),
                    "success": False
                })

        # Generate summary report
        self.generate_summary_report()

    def generate_summary_report(self) -> None:
        """Generate a summary report."""
        report_file = self.reports_dir / "processing_summary.json"

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "results": self.results,
            "success_rate": {
                "pdfs_found": self.results["pdfs_found"],
                "pdfs_processed": self.results["pdfs_processed"],
                "processing_errors": self.results["processing_errors"]
            }
        }

        # Calculate success rate
        if self.results["pdfs_found"] > 0:
            summary["success_rate"]["processing_success_rate"] = (
                self.results["pdfs_processed"] / self.results["pdfs_found"]
            )

        # Save report
        import json
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        print(f"PDFs found: {self.results['pdfs_found']}")
        print(f"PDFs processed: {self.results['pdfs_processed']}")
        print(f"Processing errors: {self.results['processing_errors']}")

        if self.results["pdfs_found"] > 0:
            success_rate = self.results["pdfs_processed"] / self.results["pdfs_found"]
            print(f"Success rate: {success_rate:.1%}")

        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"📊 Summary report: {report_file}")
        print("="*70)


def main():
    """Main execution function."""
    # Initialize processor
    processor = RealPDFProcessor()

    # Process PDFs
    processor.process_all_pdfs()


if __name__ == "__main__":
    main()
