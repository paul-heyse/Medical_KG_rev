#!/usr/bin/env python3
"""Test MinerU directly using its Python CLI interface.

This script demonstrates direct MinerU processing without going through
the wrapper layer, using the MinerU Python CLI interface.

Usage:
    python test_mineru_direct.py

Requirements:
    - MinerU package installed (mineru>=2.5.4)
    - PDF files in random_papers_output/pdfs/
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_mineru_direct():
    """Test MinerU directly using its Python CLI interface."""
    print("🧪 Testing MinerU directly...")

    try:
        # Import MinerU CLI
        from mineru.cli.client import main as mineru_main
        print("✅ MinerU CLI imported successfully")

        # Find PDF files
        pdf_dir = Path("random_papers_output/pdfs")
        if not pdf_dir.exists():
            print("❌ No PDF directory found")
            return False

        pdfs = list(pdf_dir.glob("*.pdf"))
        if not pdfs:
            print("❌ No PDF files found")
            return False

        print(f"📁 Found {len(pdfs)} PDF files")

        # Create output directory
        output_dir = Path("mineru_direct_output")
        output_dir.mkdir(exist_ok=True)

        # Process each PDF
        results = []
        for pdf_file in pdfs:
            print(f"🔄 Processing {pdf_file.name}...")

            # Set up command line arguments for MinerU
            original_argv = sys.argv.copy()
            sys.argv = [
                "mineru",
                "--path", str(pdf_file),
                "--output", str(output_dir),
                "--method", "auto",
                "--backend", "pipeline",
                "--device", "cuda",
                "--formula", "true",
                "--table", "true"
            ]

            try:
                start_time = time.time()
                mineru_main()
                processing_time = time.time() - start_time

                # Check for output files
                output_files = list(output_dir.glob(f"**/{pdf_file.stem}*"))

                result = {
                    "pdf_file": str(pdf_file),
                    "processing_time": processing_time,
                    "output_files": [str(f) for f in output_files],
                    "success": len(output_files) > 0
                }

                if result["success"]:
                    print(f"✅ Successfully processed {pdf_file.name}")
                    print(f"   Processing time: {processing_time:.2f}s")
                    print(f"   Output files: {len(output_files)}")
                    for output_file in output_files:
                        print(f"     {output_file}")
                else:
                    print(f"❌ No output files generated for {pdf_file.name}")

                results.append(result)

            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")
                results.append({
                    "pdf_file": str(pdf_file),
                    "error": str(e),
                    "success": False
                })
            finally:
                # Restore original argv
                sys.argv = original_argv

        # Summary
        successful = sum(1 for r in results if r["success"])
        total = len(results)

        print("\n" + "="*70)
        print("MINERU DIRECT PROCESSING SUMMARY")
        print("="*70)
        print(f"Total PDFs: {total}")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {successful/total:.1%}" if total > 0 else "N/A")

        if successful > 0:
            print(f"\n📁 Output directory: {output_dir}")
            print("✅ MinerU direct processing successful!")
        else:
            print("❌ MinerU direct processing failed")

        return successful > 0

    except ImportError as e:
        print(f"❌ Failed to import MinerU CLI: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main execution function."""
    print("="*70)
    print("MINERU DIRECT PROCESSING TEST")
    print("="*70)

    success = test_mineru_direct()

    print("="*70)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
