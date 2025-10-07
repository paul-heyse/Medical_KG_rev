# API Authorizations

This repo integrates with multiple public biomedical APIs. Use this checklist to capture sign-ups, keys, and storage practices needed before running ingestion end-to-end.

## Secrets Management Defaults

- Store long-lived credentials in Vault when available (default path `security/api-keys`) or in environment variables for local work (`src/Medical_KG_rev/config/settings.py:201-228`).
- Secret records support metadata such as rotation timestamps; populate `rotated_at` when you refresh keys (`src/Medical_KG_rev/config/settings.py:188-209`).
- Keep user-agent contact emails and OAuth client metadata alongside keys so adapters can assemble compliant headers (`1) docs/System Architecture & Design Rationale.md:1714-1716`).

## Clinical Research & Trials

| Provider | Auth Mechanism | Required Actions | Notes |
| --- | --- | --- | --- |
| ClinicalTrials.gov API v2 | None | No signup needed; stay within published rate limits. | Public endpoint, 10 req/sec burst (`1) docs/System Architecture & Design Rationale.md:1674-1681`). |
| OpenFDA (drug labels, adverse events, devices) | Optional API key | Request an API key from openFDA if you expect >1000 calls/day; store as `OPENFDA_API_KEY` or via Vault. | Default quota without key plus higher tiers noted in `1) docs/System Architecture & Design Rationale.md:1683`. |

## Scholarly Literature & Open Access

| Provider | Auth Mechanism | Required Actions | Notes |
| --- | --- | --- | --- |
| OpenAlex | Email-based polite pool | Register a contact email with OpenAlex and configure adapters to send `User-Agent` with `mailto`. | Needed for the 100k/day polite pool (`1) docs/System Architecture & Design Rationale.md:1682`, `1) docs/System Architecture & Design Rationale.md:1714-1716`). |
| Unpaywall | Contact email query param | Supply a valid email in adapter settings; no API key required. | Email is mandatory for higher rate limits (`1) docs/System Architecture & Design Rationale.md:1684`, `1) docs/System Architecture & Design Rationale.md:1714-1716`). |
| Crossref | Email + optional Plus token | Join Crossref Plus if you need >50 req/sec; store the token as `CROSSREF_PLUS_TOKEN`. | Plus tier documented in `1) docs/System Architecture & Design Rationale.md:1685`. |
| Europe PMC | None | No signup required; monitor rate limits. | SOAP/REST support noted in `1) docs/System Architecture & Design Rationale.md:1686`; adapter implementation at `src/Medical_KG_rev/adapters/biomedical.py:614`. |
| PubMed Central | None | No signup required; include contact email for courtesy limits if desired. | Covered by the Europe PMC adapter (`src/Medical_KG_rev/adapters/biomedical.py:614-676`). |
| CORE | API key | Create an account at CORE and generate an API key; store as `CORE_API_KEY`. | Key required for PDF access (`1) docs/System Architecture & Design Rationale.md:1688`). |
| Semantic Scholar | API key | Request a key from Semantic Scholar and store as `SEMANTIC_SCHOLAR_API_KEY`. | Requirement shown in table and config example (`1) docs/System Architecture & Design Rationale.md:1687`, `1) docs/System Architecture & Design Rationale.md:1711-1716`). |

## Drug Safety & Ontologies

| Provider | Auth Mechanism | Required Actions | Notes |
| --- | --- | --- | --- |
| RxNorm (NLM) | None | No signup required; observe rate caps. | Public API referenced at `1) docs/System Architecture & Design Rationale.md:1691`. |
| ICD-11 WHO API | OAuth 2.0 client credentials | Apply for WHO ICD API access to obtain `client_id`/`client_secret`; store as `ICD11_CLIENT_ID` and `ICD11_CLIENT_SECRET`. | OAuth workflow outlined in `1) docs/System Architecture & Design Rationale.md:1690-1710`. |
| MeSH (NLM) | None | No signup required. | Adapter uses public descriptor endpoints (`src/Medical_KG_rev/adapters/biomedical.py:763-806`). |
| ChEMBL | None | No signup required; follow EBI rate limits. | Rate limit note at `1) docs/System Architecture & Design Rationale.md:1689`; adapter implementation at `src/Medical_KG_rev/adapters/biomedical.py:824-886`. |

## Storage & Rotation Checklist

1. Create Vault entries (or local env vars) for each key-based provider before deployment (`src/Medical_KG_rev/config/settings.py:201-228`).
2. Capture contact emails for OpenAlex, Unpaywall, and Crossref in the same secret bundle so adapters can build compliant headers (`1) docs/System Architecture & Design Rationale.md:1714-1716`).
3. Schedule regular rotation for Semantic Scholar, CORE, OpenFDA, and ICD-11 credentials; update the `rotated_at` metadata when done (`src/Medical_KG_rev/config/settings.py:188-209`).
4. Document who owns each external account to avoid orphaned credentials during incident response (`openspec/project.md:520`).
