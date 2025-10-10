#!/usr/bin/env python3
"""Test MinerU simulation mode for PDF processing."""
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest


def test_mineru_simulation():
    """Test MinerU in simulation mode."""
    print("🧪 Testing MinerU simulation mode...")

    try:
        # Create a simple PDF-like content for testing
        test_content = b"Test PDF content for MinerU simulation"

        # Initialize MinerU processor
        processor = MineruProcessor()

        # Create request
        request = MineruRequest(tenant_id="test", document_id="test-doc-1", content=test_content)

        print("🔄 Processing test document...")
        response = processor.process(request)

        print("✅ Processing successful!")
        print(f"   Document ID: {response.document.document_id if response.document else 'None'}")
        print(f"   Blocks: {len(response.document.blocks) if response.document else 0}")
        print(f"   Metadata: {response.metadata.worker_id if response.metadata else 'None'}")

        return True

    except Exception as e:
        print(f"❌ MinerU simulation test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_mineru_simulation()
    sys.exit(0 if success else 1)
