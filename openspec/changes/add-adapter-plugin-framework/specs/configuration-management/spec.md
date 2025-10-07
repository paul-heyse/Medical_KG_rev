# Configuration Management Specification Deltas

## MODIFIED Requirements

### Requirement: Configuration Schema

All configuration MUST use Pydantic models with pydantic-settings integration. Configuration sources in priority order: environment variables → `.env` file → defaults. Sensitive values MUST use `SecretStr` type.

#### Scenario: Configuration precedence

- **GIVEN** default `timeout_seconds=30` in Pydantic model
- **AND** `.env` file sets `MK_ADAPTER_TIMEOUT_SECONDS=60`
- **AND** environment variable `MK_ADAPTER_TIMEOUT_SECONDS=120`
- **WHEN** configuration is loaded
- **THEN** `settings.timeout_seconds` equals 120 (environment wins)

#### Scenario: Secret string protection

- **GIVEN** adapter configuration with `api_key: SecretStr`
- **WHEN** configuration is logged or serialized
- **THEN** the API key value is masked as "**********"
- **AND** the actual value is only accessible via `get_secret_value()`

## ADDED Requirements

### Requirement: Secret Management

Sensitive adapter configuration (API keys, tokens, passwords) SHALL be stored in HashiCorp Vault or environment variables, never in code or configuration files. The system SHALL provide `VaultSecretProvider` class for loading secrets at runtime with automatic rotation support.

#### Scenario: Vault secret loading

- **GIVEN** Vault server at `https://vault.example.com`
- **AND** secret stored at path `secret/adapters/clinicaltrials` with key `api_key`
- **WHEN** adapter configuration is loaded
- **THEN** `VaultSecretProvider` fetches secret from Vault
- **AND** secret is cached for 15 minutes
- **AND** secret is automatically refreshed on expiration

#### Scenario: Vault connection failure

- **GIVEN** Vault server is unreachable
- **WHEN** adapter attempts to load configuration
- **THEN** system falls back to environment variables
- **AND** warning is logged about Vault unavailability
- **AND** application continues with env-based secrets

### Requirement: Environment Variable Conventions

All adapter configuration environment variables SHALL use the prefix `MK_ADAPTER_` followed by the setting name in uppercase. Adapter-specific settings SHALL use format `MK_ADAPTER_<ADAPTER_NAME>_<SETTING>`.

#### Scenario: Standard environment variable naming

- **GIVEN** adapter configuration for timeout setting
- **WHEN** environment variable is defined
- **THEN** variable name is `MK_ADAPTER_TIMEOUT_SECONDS`
- **AND** value is parsed as integer
- **AND** invalid values raise ValidationError on startup

#### Scenario: Adapter-specific configuration

- **GIVEN** ClinicalTrials adapter needs API key
- **WHEN** environment variable is defined
- **THEN** variable name is `MK_ADAPTER_CLINICALTRIALS_API_KEY`
- **AND** value is stored as SecretStr
- **AND** value is available only to ClinicalTrials adapter

### Requirement: Configuration Hot-Reload

The system SHALL support hot-reloading of non-sensitive adapter configuration without restart. Configuration changes SHALL be detected within 30 seconds and applied to new requests. Active requests SHALL continue with old configuration.

#### Scenario: Configuration update without restart

- **GIVEN** adapter with `timeout_seconds=30`
- **WHEN** environment variable is changed to `MK_ADAPTER_TIMEOUT_SECONDS=60`
- **AND** configuration refresh is triggered
- **THEN** new requests use timeout of 60 seconds
- **AND** in-flight requests continue with 30 seconds
- **AND** no requests are dropped during configuration change

#### Scenario: Invalid configuration update

- **GIVEN** adapter configuration hot-reload
- **WHEN** new configuration has validation error
- **THEN** error is logged with details
- **AND** old configuration remains active
- **AND** alert is triggered for operations team
