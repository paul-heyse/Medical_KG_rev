# PDF Ingestion Orchestration Notes

Adapters that expose PDF capabilities now emit a normalised `pdf_manifest`
within each `Document`. The orchestration layers continue to read
`metadata["pdf_urls"]` for compatibility but the field is populated from the
manifest to guarantee ordering and deduplication.

## Connector polite headers

The download stages should reference the adapter-supplied polite headers when
constructing HTTP clients. These headers are available both in the manifest and
via the adapter's `polite_headers()` helper, allowing stage builders to inject
connector-specific contact emails and user agents into outbound requests.

## Storage-aware download stage expectations

- Receives a list of URLs sourced from the manifest (`PipelineState.metadata["pdf_urls"]`).
- Applies connector-level rate limiting and timeout values derived from
  `ConnectorPdfSettings`.
- Persists the manifest metadata alongside storage receipts so the gate stages
  can confirm download success before releasing MinerU tasks.

## Ledger updates

When downloads complete, ledger entries should include the storage URI,
checksum, and connector identifier taken from the manifest. This metadata makes
it possible to trace failures back to the originating adapter and polite pool
configuration.
