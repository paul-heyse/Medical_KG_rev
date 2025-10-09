## MODIFIED Requirements

### Requirement: PDF Two-Phase Pipeline
The platform SHALL route OpenAlex ingest requests through the pdf-two-phase pipeline and supply adapter parameters needed for polite pool compliance.

#### Scenario: Routing OpenAlex dataset
- **GIVEN** the gateway receives an ingest request with `dataset=openalex`
- **WHEN** the pipeline is resolved
- **THEN** the `pdf-two-phase` topology SHALL be selected
- **AND** the ingest stage SHALL use the `openalex` adapter

#### Scenario: Injecting adapter parameters
- **GIVEN** OpenAlex polite pool settings are configured
- **WHEN** an ingest job is submitted
- **THEN** the adapter request SHALL include `contact_email`, `user_agent`, `max_results`, `requests_per_second`, and `timeout_seconds`
- **AND** these values SHALL be available to the adapter during execution
