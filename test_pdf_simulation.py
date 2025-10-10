#!/usr/bin/env python3
"""Test PDF processing simulation with text content."""
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest


def test_pdf_simulation():
    """Test PDF processing simulation with text content."""
    print("🧪 Testing PDF processing simulation...")

    try:
        # Create simulated PDF content as text
        simulated_pdf_content = b"""
        Title: Machine Learning in Medical Diagnosis

        Abstract:
        This paper presents a comprehensive review of machine learning applications
        in medical diagnosis. We explore various algorithms and their effectiveness
        in different medical domains.

        Introduction:
        Machine learning has revolutionized medical diagnosis by providing
        automated tools for pattern recognition and decision support.

        Methods:
        We conducted a systematic review of 100+ papers published between
        2020-2024, focusing on deep learning approaches for medical imaging.

        Results:
        Our analysis shows that convolutional neural networks achieve
        95% accuracy in medical image classification tasks.

        Conclusion:
        Machine learning continues to show promise in medical diagnosis,
        with significant improvements in accuracy and efficiency.

        References:
        1. Smith, J. et al. (2023). Deep Learning in Medical Imaging.
        2. Johnson, A. et al. (2022). AI Applications in Healthcare.
        """

        # Initialize MinerU processor
        processor = MineruProcessor()

        # Create request
        request = MineruRequest(
            tenant_id="test", document_id="simulated-pdf-1", content=simulated_pdf_content
        )

        print("🔄 Processing simulated PDF content...")
        response = processor.process(request)

        print("✅ Processing successful!")
        print(f"   Document ID: {response.document.document_id if response.document else 'None'}")
        print(f"   Blocks: {len(response.document.blocks) if response.document else 0}")
        print(f"   Tables: {len(response.document.tables) if response.document else 0}")
        print(f"   Figures: {len(response.document.figures) if response.document else 0}")
        print(f"   Worker ID: {response.metadata.worker_id if response.metadata else 'None'}")

        # Show sample blocks
        if response.document and response.document.blocks:
            print("\n📄 Sample blocks:")
            for i, block in enumerate(response.document.blocks[:3]):
                print(
                    f"   Block {i+1}: {block.text[:100]}..."
                    if block.text
                    else f"   Block {i+1}: [No text]"
                )

        return True

    except Exception as e:
        print(f"❌ PDF simulation test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_pdf_simulation()
    sys.exit(0 if success else 1)
