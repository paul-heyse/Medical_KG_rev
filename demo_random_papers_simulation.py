#!/usr/bin/env python3
"""Demo: Download random papers and simulate PDF processing with MinerU.

This script demonstrates the complete workflow:
1. Fetch random papers from OpenAlex using pyalex
2. Download PDFs for papers that have them
3. Simulate PDF processing using MinerU with text content
4. Generate reports on the processing results

Usage:
    python demo_random_papers_simulation.py

Environment Variables:
    PYALEX_EMAIL: Email address for pyalex API access (default: paul@heyse.io)
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import requests

from Medical_KG_rev.adapters.base import AdapterContext
from Medical_KG_rev.adapters.openalex import OpenAlexAdapter
from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest


class RandomPaperSimulationDemo:
    """Demo for downloading and simulating PDF processing of random papers."""

    def __init__(self, sample_size: int = 20, output_dir: str = "demo_output"):
        """Initialize the demo processor."""
        self.sample_size = sample_size
        self.output_dir = Path(output_dir)
        self.pdf_dir = self.output_dir / "pdfs"
        self.processed_dir = self.output_dir / "processed"
        self.reports_dir = self.output_dir / "reports"

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.pdf_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        # Initialize OpenAlex adapter
        self.email = os.getenv("PYALEX_EMAIL", "paul@heyse.io")
        self.adapter = OpenAlexAdapter(contact_email=self.email)

        # Initialize MinerU processor
        self.mineru_processor = MineruProcessor()

        # Results tracking
        self.results = {
            "papers_fetched": 0,
            "papers_with_pdfs": 0,
            "pdfs_downloaded": 0,
            "pdfs_processed": 0,
            "processing_errors": 0,
            "processing_details": [],
        }

    def fetch_random_papers(self) -> list[dict[str, Any]]:
        """Fetch random papers from OpenAlex."""
        print(f"🔍 Fetching {self.sample_size} random papers from OpenAlex...")

        try:
            # Use OpenAlex adapter to fetch papers with broad medical query
            context = AdapterContext(
                tenant_id="demo-papers",
                domain="research",
                correlation_id="demo-fetch-1",
                parameters={"query": "machine learning medical diagnosis"},
            )

            result = self.adapter.run(context)
            documents = result.documents

            if documents:
                # Take first N documents as our "random" sample
                selected_docs = documents[: self.sample_size]

                # Convert documents to paper format
                papers = []
                for doc in selected_docs:
                    paper = {
                        "id": doc.metadata.get("openalex_id", doc.id),
                        "title": doc.title,
                        "doi": doc.metadata.get("doi"),
                        "pdf_urls": doc.metadata.get("pdf_urls", []),
                        "pdf_manifest": doc.metadata.get("pdf_manifest"),
                        "authorships": doc.metadata.get("authorships", []),
                        "publication_year": doc.metadata.get("publication_year"),
                        "is_open_access": doc.metadata.get("is_open_access", False),
                        "abstract": (
                            doc.sections[0].blocks[0].text
                            if doc.sections and doc.sections[0].blocks
                            else ""
                        ),
                        "document": doc,
                    }
                    papers.append(paper)

                self.results["papers_fetched"] = len(papers)
                print(f"✅ Fetched {len(papers)} papers")
                return papers
            else:
                print("❌ No papers found")
                return []

        except Exception as e:
            print(f"❌ Error fetching papers: {e}")
            return []

    def download_pdf(self, paper: dict[str, Any]) -> Path | None:
        """Download PDF for a paper if available."""
        pdf_urls = paper.get("pdf_urls", [])
        if not pdf_urls:
            return None

        # Try the first available PDF URL
        pdf_url = pdf_urls[0]
        paper_id = paper["id"].split("/")[-1] if "/" in paper["id"] else paper["id"]
        pdf_filename = f"{paper_id}.pdf"
        pdf_path = self.pdf_dir / pdf_filename

        try:
            print(f"📥 Downloading PDF: {pdf_filename}")

            # Use requests to download the PDF
            headers = {"User-Agent": f"Medical_KG_rev/1.0 ({self.email})", "From": self.email}

            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()

            # Save the PDF
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            print(f"✅ Downloaded: {pdf_filename} ({len(response.content)} bytes)")
            return pdf_path

        except Exception as e:
            print(f"❌ Failed to download PDF for {paper_id}: {e}")
            return None

    def simulate_pdf_processing(self, pdf_path: Path, paper: dict[str, Any]) -> dict[str, Any]:
        """Simulate PDF processing using MinerU with text content."""
        try:
            print(f"🔄 Simulating PDF processing: {pdf_path.name}")

            # Create simulated PDF content based on paper metadata
            simulated_content = f"""
            Title: {paper['title']}

            Authors: {', '.join(paper.get('authorships', [])[:3])}

            Publication Year: {paper.get('publication_year', 'Unknown')}

            DOI: {paper.get('doi', 'N/A')}

            Abstract:
            {paper.get('abstract', 'Abstract not available')}

            Introduction:
            This paper presents research in the field of medical machine learning.
            The study focuses on applications of artificial intelligence in healthcare
            and medical diagnosis.

            Methods:
            The research methodology includes data collection, preprocessing,
            model training, and evaluation using various machine learning algorithms.

            Results:
            The experimental results demonstrate significant improvements in
            medical diagnosis accuracy and efficiency.

            Discussion:
            The findings contribute to the growing body of research in
            medical AI applications and provide insights for future work.

            Conclusion:
            This study advances the field of medical machine learning and
            demonstrates the potential for AI-assisted healthcare.

            References:
            1. Smith, J. et al. (2023). Machine Learning in Medicine.
            2. Johnson, A. et al. (2022). AI Applications in Healthcare.
            3. Brown, M. et al. (2021). Deep Learning for Medical Diagnosis.
            """.encode()

            # Create a clean document ID for MinerU
            clean_doc_id = pdf_path.stem

            # Create MinerU request
            request = MineruRequest(
                tenant_id="demo-papers", document_id=clean_doc_id, content=simulated_content
            )

            # Process with MinerU
            start_time = time.time()
            response = self.mineru_processor.process(request)
            processing_time = time.time() - start_time

            # Save processed results
            paper_id = paper["id"].split("/")[-1] if "/" in paper["id"] else paper["id"]
            output_file = self.processed_dir / f"{paper_id}_processed.json"

            # Convert response to serializable format
            processed_data = {
                "paper_id": paper["id"],
                "title": paper["title"],
                "doi": paper.get("doi"),
                "processing_time_seconds": processing_time,
                "document_blocks": len(response.document.blocks) if response.document else 0,
                "document_id": response.document.document_id if response.document else None,
                "tables": len(response.document.tables) if response.document else 0,
                "figures": len(response.document.figures) if response.document else 0,
                "equations": len(response.document.equations) if response.document else 0,
                "processing_metadata": {
                    "worker_id": response.metadata.worker_id if response.metadata else None,
                    "processing_time_ms": (
                        response.metadata.processing_time_ms if response.metadata else None
                    ),
                    "gpu_utilization": (
                        response.metadata.gpu_utilization if response.metadata else None
                    ),
                },
                "sample_blocks": [
                    {
                        "text": (
                            block.text[:200] + "..."
                            if block.text and len(block.text) > 200
                            else block.text
                        ),
                        "page": block.page,
                        "kind": block.kind,
                    }
                    for block in (response.document.blocks[:3] if response.document else [])
                ],
                "success": True,
            }

            # Save to JSON file
            import json

            with open(output_file, "w") as f:
                json.dump(processed_data, f, indent=2)

            print(f"✅ Simulated processing: {pdf_path.name} ({processing_time:.2f}s)")
            print(f"   Blocks: {processed_data['document_blocks']}")
            print(f"   Tables: {processed_data['tables']}")
            print(f"   Figures: {processed_data['figures']}")

            return processed_data

        except Exception as e:
            print(f"❌ Error simulating PDF processing {pdf_path.name}: {e}")
            return {
                "paper_id": paper["id"],
                "title": paper["title"],
                "error": str(e),
                "success": False,
            }

    def process_all_papers(self) -> None:
        """Main processing workflow."""
        print("=" * 70)
        print("RANDOM PAPERS DOWNLOAD AND SIMULATION DEMO")
        print("=" * 70)
        print(f"Sample size: {self.sample_size}")
        print(f"Output directory: {self.output_dir}")
        print(f"Email: {self.email}")
        print("-" * 70)

        # Step 1: Fetch papers
        papers = self.fetch_random_papers()
        if not papers:
            print("❌ No papers to process")
            return

        # Step 2: Identify papers with PDFs
        papers_with_pdfs = [p for p in papers if p.get("pdf_urls")]
        self.results["papers_with_pdfs"] = len(papers_with_pdfs)

        print(f"📊 Papers with PDFs: {len(papers_with_pdfs)}/{len(papers)}")

        if not papers_with_pdfs:
            print("⚠️  No papers with PDFs found")
            return

        # Step 3: Download PDFs
        print("\n📥 Downloading PDFs...")
        downloaded_pdfs = []
        for paper in papers_with_pdfs:
            pdf_path = self.download_pdf(paper)
            if pdf_path:
                downloaded_pdfs.append((pdf_path, paper))
                self.results["pdfs_downloaded"] += 1

        print(f"✅ Downloaded {len(downloaded_pdfs)} PDFs")

        if not downloaded_pdfs:
            print("⚠️  No PDFs downloaded successfully")
            return

        # Step 4: Simulate PDF processing with MinerU
        print("\n🔄 Simulating PDF processing with MinerU...")
        for pdf_path, paper in downloaded_pdfs:
            try:
                result = self.simulate_pdf_processing(pdf_path, paper)
                self.results["processing_details"].append(result)

                if result["success"]:
                    self.results["pdfs_processed"] += 1
                else:
                    self.results["processing_errors"] += 1

            except Exception as e:
                print(f"❌ Unexpected error processing {pdf_path.name}: {e}")
                self.results["processing_errors"] += 1
                self.results["processing_details"].append(
                    {
                        "paper_id": paper["id"],
                        "title": paper["title"],
                        "error": str(e),
                        "success": False,
                    }
                )

        # Step 5: Generate summary report
        self.generate_summary_report()

    def generate_summary_report(self) -> None:
        """Generate a summary report of the processing results."""
        report_file = self.reports_dir / "demo_summary.json"

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "demo_type": "PDF Processing Simulation",
            "sample_size": self.sample_size,
            "email": self.email,
            "results": self.results,
            "success_rate": {
                "papers_fetched": self.results["papers_fetched"],
                "papers_with_pdfs": self.results["papers_with_pdfs"],
                "pdfs_downloaded": self.results["pdfs_downloaded"],
                "pdfs_processed": self.results["pdfs_processed"],
                "processing_errors": self.results["processing_errors"],
            },
        }

        # Calculate success rates
        if self.results["papers_fetched"] > 0:
            summary["success_rate"]["pdf_availability_rate"] = (
                self.results["papers_with_pdfs"] / self.results["papers_fetched"]
            )

        if self.results["pdfs_downloaded"] > 0:
            summary["success_rate"]["processing_success_rate"] = (
                self.results["pdfs_processed"] / self.results["pdfs_downloaded"]
            )

        # Save report
        import json

        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 70)
        print("DEMO SUMMARY")
        print("=" * 70)
        print(f"Papers fetched: {self.results['papers_fetched']}")
        print(f"Papers with PDFs: {self.results['papers_with_pdfs']}")
        print(f"PDFs downloaded: {self.results['pdfs_downloaded']}")
        print(f"PDFs processed (simulated): {self.results['pdfs_processed']}")
        print(f"Processing errors: {self.results['processing_errors']}")

        if self.results["papers_fetched"] > 0:
            pdf_rate = self.results["papers_with_pdfs"] / self.results["papers_fetched"]
            print(f"PDF availability rate: {pdf_rate:.1%}")

        if self.results["pdfs_downloaded"] > 0:
            process_rate = self.results["pdfs_processed"] / self.results["pdfs_downloaded"]
            print(f"Simulation success rate: {process_rate:.1%}")

        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"📊 Summary report: {report_file}")
        print("=" * 70)


def main():
    """Main execution function."""
    # Ensure email is set
    if not os.getenv("PYALEX_EMAIL"):
        os.environ["PYALEX_EMAIL"] = "paul@heyse.io"

    # Initialize demo processor
    demo = RandomPaperSimulationDemo(sample_size=20)

    # Process papers
    demo.process_all_papers()


if __name__ == "__main__":
    main()
