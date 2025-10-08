## Why

The `adapters/biomedical.py` file defines OpenAlex, Unpaywall, Crossref, and numerous other adapters in a single monolithic file. Each class reimplements HTTP fetching, response parsing, and metadata shaping despite the presence of a full adapter plugin manager. This creates maintenance challenges, code duplication, and makes it difficult to reuse common patterns like pagination, DOI normalization, and open access link resolution.

**Critical Barrier**: The OpenAlex adapter currently stops at abstract-level metadata and never pulls PDF assets or sets document_type="pdf". It also bypasses the pyalex client entirely, so there is nothing that produces a PDF payload to hand to MinerU or chunking. This prevents end-to-end testing of the PDF processing pipeline.

## What Changes

- **Split into dedicated modules**: Break the biomedical adapter monolith into separate packages (`adapters/openalex/`, `adapters/unpaywall/`, etc.)
- **Create shared mixins**: Introduce base classes and mixins for common behaviors (pagination, DOI normalization, OA link resolution)
- **Standardize interfaces**: Ensure all adapters implement consistent interfaces for metadata extraction and PDF URL discovery
- **Improve reusability**: Make it easier to reuse request logic and expose richer metadata through consistent interfaces
- **Plugin integration**: Better integrate with the existing adapter plugin manager for improved discoverability and configuration

## Impact

- **Affected specs**: `specs/biomedical-adapters/spec.md` - Adapter architecture and interface requirements
- **Affected code**:
  - `src/Medical_KG_rev/adapters/biomedical.py` - Split into multiple adapter modules
  - `src/Medical_KG_rev/adapters/plugins/` - Update plugin manager integration
  - `src/Medical_KG_rev/adapters/base.py` - Add shared mixins and base classes
- **Affected systems**: Adapter framework, biomedical data ingestion, plugin management
