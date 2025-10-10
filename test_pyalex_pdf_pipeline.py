#!/usr/bin/env python3
"""Comprehensive test script for pyalex PDF download and processing pipeline.

This script tests the complete workflow from OpenAlex query to PDF processing,
validating all components of the Medical_KG_rev PDF processing pipeline.

Usage:
    python test_pyalex_pdf_pipeline.py

Environment Variables:
    PYALEX_EMAIL: Email address for pyalex API access (default: paul@heyse.io)
    OPENALEX_CONTACT_EMAIL: Alternative email configuration
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

from Medical_KG_rev.adapters.base import AdapterContext
from Medical_KG_rev.adapters.openalex import OpenAlexAdapter
from Medical_KG_rev.models import Document

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PyalexPipelineTester:
    """Test suite for pyalex PDF processing pipeline."""

    def __init__(self):
        """Initialize the tester with proper email configuration."""
        self.email = os.getenv("PYALEX_EMAIL", "paul@heyse.io")
        self.adapter = OpenAlexAdapter(contact_email=self.email)
        self.test_results = {}

    def test_pyalex_import(self) -> bool:
        """Test that pyalex can be imported and configured."""
        try:
            from pyalex import Works
            from pyalex import config as pyalex_config
            logger.info(f"pyalex_import_success - email: {self.email}")

            # Verify email is set in pyalex config
            if pyalex_config.get("email") == self.email:
                logger.info(f"pyalex_email_configured - email: {self.email}")
                return True
            else:
                logger.warning(f"pyalex_email_not_set - expected: {self.email}, actual: {pyalex_config.get('email')}")
                return False

        except ImportError as e:
            logger.error(f"pyalex_import_failed - error: {str(e)}")
            return False

    def test_openalex_adapter_initialization(self) -> bool:
        """Test OpenAlex adapter initialization."""
        try:
            adapter = OpenAlexAdapter(contact_email=self.email)
            logger.info("openalex_adapter_initialized",
                       email=self.email,
                       adapter_name=adapter.name)
            return True
        except Exception as e:
            logger.error("openalex_adapter_init_failed", error=str(e))
            return False

    def test_doi_search(self) -> bool:
        """Test DOI-based search functionality."""
        try:
            context = AdapterContext(
                tenant_id="test-tenant",
                domain="research",
                correlation_id="test-doi-1",
                parameters={"doi": "10.1371/journal.pone.0123456"}
            )

            logger.info("testing_doi_search", doi=context.parameters["doi"])
            documents = self.adapter.fetch_and_parse(context)

            if documents:
                doc = documents[0]
                logger.info("doi_search_success",
                           document_id=doc.id,
                           title=doc.title,
                           pdf_urls=doc.metadata.get("pdf_urls", []))
                return True
            else:
                logger.warning("doi_search_no_results")
                return False

        except Exception as e:
            logger.error("doi_search_failed", error=str(e))
            return False

    def test_keyword_search(self) -> bool:
        """Test keyword-based search functionality."""
        try:
            context = AdapterContext(
                tenant_id="test-tenant",
                domain="research",
                correlation_id="test-keyword-1",
                parameters={"query": "machine learning medical diagnosis"}
            )

            logger.info("testing_keyword_search", query=context.parameters["query"])
            documents = self.adapter.fetch_and_parse(context)

            if documents:
                logger.info("keyword_search_success",
                           result_count=len(documents),
                           first_doc_id=documents[0].id)

                # Check for PDF availability
                pdf_count = sum(1 for doc in documents
                              if doc.metadata.get("pdf_urls"))
                logger.info("pdf_availability",
                           total_docs=len(documents),
                           docs_with_pdf=pdf_count)
                return True
            else:
                logger.warning("keyword_search_no_results")
                return False

        except Exception as e:
            logger.error("keyword_search_failed", error=str(e))
            return False

    def test_openalex_id_search(self) -> bool:
        """Test OpenAlex ID-based search functionality."""
        try:
            # Use a known OpenAlex work ID
            context = AdapterContext(
                tenant_id="test-tenant",
                domain="research",
                correlation_id="test-openalex-1",
                parameters={"openalex_id": "W2755950973"}
            )

            logger.info("testing_openalex_id_search",
                       openalex_id=context.parameters["openalex_id"])
            documents = self.adapter.fetch_and_parse(context)

            if documents:
                doc = documents[0]
                logger.info("openalex_id_search_success",
                           document_id=doc.id,
                           title=doc.title)
                return True
            else:
                logger.warning("openalex_id_search_no_results")
                return False

        except Exception as e:
            logger.error("openalex_id_search_failed", error=str(e))
            return False

    def test_pdf_manifest_generation(self) -> bool:
        """Test PDF manifest generation from documents."""
        try:
            context = AdapterContext(
                tenant_id="test-tenant",
                domain="research",
                correlation_id="test-pdf-manifest-1",
                parameters={"query": "COVID-19 vaccine efficacy"}
            )

            logger.info("testing_pdf_manifest_generation")
            documents = self.adapter.fetch_and_parse(context)

            if documents:
                pdf_docs = [doc for doc in documents
                           if doc.metadata.get("pdf_urls")]

                if pdf_docs:
                    doc = pdf_docs[0]
                    manifest = doc.metadata.get("pdf_manifest")

                    if manifest:
                        logger.info("pdf_manifest_generated",
                                   manifest_assets=len(manifest.get("assets", [])),
                                   connector=manifest.get("connector"))
                        return True
                    else:
                        logger.warning("pdf_manifest_missing")
                        return False
                else:
                    logger.warning("no_pdf_documents_found")
                    return False
            else:
                logger.warning("no_documents_for_pdf_test")
                return False

        except Exception as e:
            logger.error("pdf_manifest_generation_failed", error=str(e))
            return False

    async def test_pdf_download_capability(self) -> bool:
        """Test PDF download capability (without actually downloading)."""
        try:
            context = AdapterContext(
                tenant_id="test-tenant",
                domain="research",
                correlation_id="test-pdf-download-1",
                parameters={"query": "open access medical research"}
            )

            logger.info("testing_pdf_download_capability")
            documents = self.adapter.fetch_and_parse(context)

            if documents:
                pdf_docs = [doc for doc in documents
                           if doc.metadata.get("pdf_urls")]

                if pdf_docs:
                    doc = pdf_docs[0]
                    pdf_urls = doc.metadata.get("pdf_urls", [])

                    logger.info("pdf_download_urls_found",
                               document_id=doc.id,
                               pdf_url_count=len(pdf_urls),
                               sample_url=pdf_urls[0] if pdf_urls else None)

                    # Test the download method exists and is callable
                    if hasattr(self.adapter, 'fetch_and_upload_pdf'):
                        logger.info("pdf_download_method_available")
                        return True
                    else:
                        logger.warning("pdf_download_method_missing")
                        return False
                else:
                    logger.warning("no_pdf_urls_found")
                    return False
            else:
                logger.warning("no_documents_for_download_test")
                return False

        except Exception as e:
            logger.error("pdf_download_capability_test_failed", error=str(e))
            return False

    def test_metadata_extraction(self) -> bool:
        """Test comprehensive metadata extraction."""
        try:
            context = AdapterContext(
                tenant_id="test-tenant",
                domain="research",
                correlation_id="test-metadata-1",
                parameters={"query": "artificial intelligence healthcare"}
            )

            logger.info("testing_metadata_extraction")
            documents = self.adapter.fetch_and_parse(context)

            if documents:
                doc = documents[0]
                metadata = doc.metadata

                # Check for key metadata fields
                required_fields = [
                    "openalex_id", "title", "publication_year",
                    "authorships", "concepts"
                ]

                missing_fields = [field for field in required_fields
                                if field not in metadata]

                if not missing_fields:
                    logger.info("metadata_extraction_success",
                               openalex_id=metadata.get("openalex_id"),
                               title=metadata.get("title"),
                               year=metadata.get("publication_year"),
                               author_count=len(metadata.get("authorships", [])),
                               concept_count=len(metadata.get("concepts", [])))
                    return True
                else:
                    logger.warning("metadata_fields_missing",
                                  missing_fields=missing_fields)
                    return False
            else:
                logger.warning("no_documents_for_metadata_test")
                return False

        except Exception as e:
            logger.error("metadata_extraction_failed", error=str(e))
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("starting_pyalex_pipeline_tests", email=self.email)

        tests = [
            ("pyalex_import", self.test_pyalex_import),
            ("openalex_adapter_init", self.test_openalex_adapter_initialization),
            ("doi_search", self.test_doi_search),
            ("keyword_search", self.test_keyword_search),
            ("openalex_id_search", self.test_openalex_id_search),
            ("pdf_manifest_generation", self.test_pdf_manifest_generation),
            ("metadata_extraction", self.test_metadata_extraction),
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                logger.info("running_test", test_name=test_name)
                result = test_func()
                results[test_name] = result
                status = "PASSED" if result else "FAILED"
                logger.info("test_completed", test_name=test_name, status=status)
            except Exception as e:
                logger.error("test_error", test_name=test_name, error=str(e))
                results[test_name] = False

        # Run async test
        try:
            logger.info("running_async_test", test_name="pdf_download_capability")
            result = asyncio.run(self.test_pdf_download_capability())
            results["pdf_download_capability"] = result
            status = "PASSED" if result else "FAILED"
            logger.info("async_test_completed", test_name="pdf_download_capability", status=status)
        except Exception as e:
            logger.error("async_test_error", test_name="pdf_download_capability", error=str(e))
            results["pdf_download_capability"] = False

        return results

    def print_summary(self, results: Dict[str, bool]) -> None:
        """Print test results summary."""
        print("\n" + "="*60)
        print("PYALEX PDF PIPELINE TEST RESULTS")
        print("="*60)
        print(f"Email Configuration: {self.email}")
        print(f"Total Tests: {len(results)}")

        passed = sum(1 for result in results.values() if result)
        failed = len(results) - passed

        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print("-"*60)

        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:30} {status}")

        print("-"*60)

        if failed == 0:
            print("🎉 ALL TESTS PASSED! Ready for full PDF processing pipeline.")
        else:
            print("⚠️  Some tests failed. Review logs for details.")

        print("="*60)


def main():
    """Main test execution function."""
    # Ensure email is set
    if not os.getenv("PYALEX_EMAIL"):
        os.environ["PYALEX_EMAIL"] = "paul@heyse.io"

    tester = PyalexPipelineTester()
    results = tester.run_all_tests()
    tester.print_summary(results)

    # Exit with appropriate code
    failed_tests = sum(1 for result in results.values() if not result)
    sys.exit(failed_tests)


if __name__ == "__main__":
    main()
