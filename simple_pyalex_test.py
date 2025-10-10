#!/usr/bin/env python3
"""Simple test script for pyalex PDF download and processing pipeline."""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_pyalex_import():
    """Test that pyalex can be imported and configured."""
    try:
        from pyalex import config as pyalex_config

        print("✅ pyalex import successful")

        # Set email if not already set
        email = os.getenv("PYALEX_EMAIL", "paul@heyse.io")
        pyalex_config["email"] = email
        print(f"✅ pyalex email configured: {email}")
        return True
    except ImportError as e:
        print(f"❌ pyalex import failed: {e}")
        return False


def test_openalex_adapter():
    """Test OpenAlex adapter initialization."""
    try:
        from Medical_KG_rev.adapters.openalex import OpenAlexAdapter

        adapter = OpenAlexAdapter(contact_email="paul@heyse.io")
        print(f"✅ OpenAlex adapter initialized: {adapter.name}")
        return True
    except Exception as e:
        print(f"❌ OpenAlex adapter initialization failed: {e}")
        return False


def test_basic_search():
    """Test basic search functionality."""
    try:
        from Medical_KG_rev.adapters.base import AdapterContext
        from Medical_KG_rev.adapters.openalex import OpenAlexAdapter

        adapter = OpenAlexAdapter(contact_email="paul@heyse.io")
        context = AdapterContext(
            tenant_id="test-tenant",
            domain="research",
            correlation_id="test-1",
            parameters={"query": "machine learning medical diagnosis"},
        )

        print("🔍 Testing keyword search...")
        result = adapter.run(context)
        documents = result.documents

        if documents:
            doc = documents[0]
            print(f"✅ Search successful - found {len(documents)} documents")
            print(f"   First document ID: {doc.id}")
            print(f"   Title: {doc.title}")

            # Check for PDF availability
            pdf_urls = doc.metadata.get("pdf_urls", [])
            if pdf_urls:
                print(f"   PDF URLs found: {len(pdf_urls)}")
                print(f"   Sample PDF URL: {pdf_urls[0]}")
            else:
                print("   No PDF URLs found")

            return True
        else:
            print("❌ No documents found")
            return False

    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


def test_doi_search():
    """Test DOI-based search."""
    try:
        from Medical_KG_rev.adapters.base import AdapterContext
        from Medical_KG_rev.adapters.openalex import OpenAlexAdapter

        adapter = OpenAlexAdapter(contact_email="paul@heyse.io")
        context = AdapterContext(
            tenant_id="test-tenant",
            domain="research",
            correlation_id="test-doi-1",
            parameters={"doi": "10.1371/journal.pone.0123456"},
        )

        print("🔍 Testing DOI search...")
        result = adapter.run(context)
        documents = result.documents

        if documents:
            doc = documents[0]
            print("✅ DOI search successful")
            print(f"   Document ID: {doc.id}")
            print(f"   Title: {doc.title}")
            print(f"   DOI: {doc.metadata.get('doi')}")
            return True
        else:
            print("❌ DOI search returned no results")
            return False

    except Exception as e:
        print(f"❌ DOI search test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("PYALEX PDF PIPELINE TEST")
    print("=" * 60)

    # Ensure email is set
    if not os.getenv("PYALEX_EMAIL"):
        os.environ["PYALEX_EMAIL"] = "paul@heyse.io"

    print(f"Email configuration: {os.getenv('PYALEX_EMAIL')}")
    print("-" * 60)

    tests = [
        ("pyalex_import", test_pyalex_import),
        ("openalex_adapter", test_openalex_adapter),
        ("basic_search", test_basic_search),
        ("doi_search", test_doi_search),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print("-" * 60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")

    print("-" * 60)

    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for full PDF processing pipeline.")
    else:
        print("⚠️  Some tests failed. Review output above for details.")

    print("=" * 60)

    # Exit with appropriate code
    sys.exit(total - passed)


if __name__ == "__main__":
    main()
