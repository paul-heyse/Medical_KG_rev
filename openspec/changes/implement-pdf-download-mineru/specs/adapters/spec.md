## MODIFIED Requirements

### Requirement: OpenAlex Adapter PDF Metadata

The OpenAlex adapter SHALL extract and include PDF URL information from the `best_oa_location.url_for_pdf` field in document metadata.

#### Scenario: PDF URL extraction

- **GIVEN** an OpenAlex record with open access PDF information
- **WHEN** the adapter processes the record
- **THEN** it extracts the `best_oa_location.url_for_pdf` URL
- **AND** includes it in the document metadata as `pdf_url`
- **AND** validates the URL is accessible and contains PDF content

#### Scenario: PDF metadata validation

- **GIVEN** extracted PDF metadata
- **WHEN** the document is created
- **THEN** PDF URLs are validated for accessibility
- **AND** PDF file size and content type are captured when available
- **AND** invalid or inaccessible PDF URLs are handled gracefully
