## ADDED Requirements

### Requirement: OpenAlex Adapter Configuration
The system SHALL expose configuration controls for the OpenAlex adapter to satisfy polite pool requirements.

#### Scenario: Configuring OpenAlex polite pool values
- **WHEN** operators define `MK_OPENALEX__*` settings (contact email, user-agent, limits)
- **THEN** the runtime configuration SHALL surface these values via `OpenAlexSettings`
- **AND** the adapter SHALL apply them when initialising the pyalex client

#### Scenario: Default configuration for local development
- **WHEN** no explicit OpenAlex settings are provided
- **THEN** sensible defaults (e.g., `oss@medical-kg.local`) SHALL be applied
- **AND** the adapter SHALL continue to operate for local testing without manual configuration
