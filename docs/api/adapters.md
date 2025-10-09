# PDF-capable Adapter Configuration

The biomedical adapters that download PDF assets share a common configuration
surface powered by the `ConnectorPdfSettings` model. Each adapter attaches a
`pdf_manifest` block to emitted `Document` metadata that includes the
standardised asset descriptors and the polite headers used for outbound
requests.

Downstream consumers should prefer the manifest over legacy `pdf_urls` fields,
though the URLs remain available for backwards compatibility during the
migration period.

## Shared settings

All connectors honour the following environment variables using the
`MK_<CONNECTOR>__PDF__*` prefix:

- `CONTACT_EMAIL` – contact address sent via the `From` header.
- `USER_AGENT` – optional user agent override for HTTP requests.
- `REQUESTS_PER_SECOND` – steady-state rate limiter budget.
- `BURST` – short-term burst allowance for the limiter.
- `TIMEOUT_SECONDS` – HTTP client timeout for download and metadata calls.
- `RETRY_ATTEMPTS` – number of retry attempts for transient failures.
- `RETRY_BACKOFF_SECONDS` – base backoff interval between retries.
- `MAX_FILE_SIZE_MB` – maximum accepted PDF size before rejection.
- `MAX_REDIRECTS` – maximum redirects followed during download resolution.

## Adapter specific guidance

### OpenAlex

```
#MK_OPENALEX__PDF__CONTACT_EMAIL=oss@example.com
#MK_OPENALEX__PDF__USER_AGENT=MedicalKG/1.0 (mailto:oss@example.com)
#MK_OPENALEX__PDF__REQUESTS_PER_SECOND=5.0
#MK_OPENALEX__PDF__BURST=5
#MK_OPENALEX__PDF__TIMEOUT_SECONDS=30.0
#MK_OPENALEX__PDF__RETRY_ATTEMPTS=3
#MK_OPENALEX__PDF__RETRY_BACKOFF_SECONDS=1.0
#MK_OPENALEX__PDF__MAX_FILE_SIZE_MB=100
#MK_OPENALEX__PDF__MAX_REDIRECTS=5
```

The OpenAlex adapter normalises manifest entries from primary and alternate
locations, deduplicating URLs by version.

### Unpaywall

```
#MK_UNPAYWALL__PDF__CONTACT_EMAIL=oa-team@example.com
#MK_UNPAYWALL__PDF__USER_AGENT=MedicalKG/1.0 (mailto:oa-team@example.com)
#MK_UNPAYWALL__PDF__REQUESTS_PER_SECOND=5.0
#MK_UNPAYWALL__PDF__BURST=5
#MK_UNPAYWALL__PDF__TIMEOUT_SECONDS=20.0
#MK_UNPAYWALL__PDF__RETRY_ATTEMPTS=3
#MK_UNPAYWALL__PDF__RETRY_BACKOFF_SECONDS=1.0
#MK_UNPAYWALL__PDF__MAX_FILE_SIZE_MB=100
#MK_UNPAYWALL__PDF__MAX_REDIRECTS=5
```

Unpaywall manifests inherit the best open-access location, capturing landing
pages, licences, versions, and host types for downstream auditing.

### Crossref

```
#MK_CROSSREF__PDF__CONTACT_EMAIL=crossref@example.com
#MK_CROSSREF__PDF__USER_AGENT=MedicalKG/1.0 (mailto:crossref@example.com)
#MK_CROSSREF__PDF__REQUESTS_PER_SECOND=4.0
#MK_CROSSREF__PDF__BURST=4
#MK_CROSSREF__PDF__TIMEOUT_SECONDS=25.0
#MK_CROSSREF__PDF__RETRY_ATTEMPTS=3
#MK_CROSSREF__PDF__RETRY_BACKOFF_SECONDS=0.5
#MK_CROSSREF__PDF__MAX_FILE_SIZE_MB=100
#MK_CROSSREF__PDF__MAX_REDIRECTS=5
```

Crossref manifests are derived from the API `link` array, preserving inferred
licences and checksum hints when available.

### Europe PMC

```
#MK_PMC__PDF__CONTACT_EMAIL=pmc@example.com
#MK_PMC__PDF__USER_AGENT=MedicalKG/1.0 (mailto:pmc@example.com)
#MK_PMC__PDF__REQUESTS_PER_SECOND=3.0
#MK_PMC__PDF__BURST=3
#MK_PMC__PDF__TIMEOUT_SECONDS=30.0
#MK_PMC__PDF__RETRY_ATTEMPTS=3
#MK_PMC__PDF__RETRY_BACKOFF_SECONDS=1.0
#MK_PMC__PDF__MAX_FILE_SIZE_MB=100
#MK_PMC__PDF__MAX_REDIRECTS=5
```

The PMC adapter extracts PDF references from XML `self-uri` and `ext-link`
entries, linking them to a canonical landing page derived from the PMCID.
