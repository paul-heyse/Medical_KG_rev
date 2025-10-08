# Configuration Spec Delta

## ADDED Requirements

### Requirement: Pydantic Configuration Models

The orchestration system MUST provide Pydantic models for loading and validating YAML configuration files. The configuration system SHALL include:

- `ResiliencePolicyConfig`: Named resilience policies with retry, backoff, circuit breaker, and rate limit settings
- `StageDefinition`: Single stage specification with name, type, policy reference, and dependencies
- `GateCondition`: Gate predicate with ledger field and expected value
- `PipelineTopologyConfig`: Complete pipeline definition with stages, gates, and metadata
- `PipelineConfigLoader`: Loader with YAML parsing, validation, and caching

All models MUST use Pydantic field validators to enforce constraints (e.g., max_attempts between 1-10, stage names alphanumeric+underscores only).

#### Scenario: Pipeline topology loaded and validated

- **GIVEN** a file `config/orchestration/pipelines/auto.yaml` with 8 stages
- **WHEN** `PipelineConfigLoader().load("auto")` is called
- **THEN** a `PipelineTopologyConfig` instance is returned
- **AND** all stages are validated (names, types, policy references)
- **AND** dependencies form a DAG (no cycles detected)
- **AND** the config is cached for subsequent loads

#### Scenario: Invalid pipeline topology rejected

- **GIVEN** a pipeline YAML with a cyclic dependency (stage A depends on B, B depends on A)
- **WHEN** loading the config
- **THEN** a `ValidationError` is raised with message "Cycle detected in pipeline"
- **AND** the specific stages involved in the cycle are identified

---

### Requirement: Resilience Policy Validation

Resilience policy configurations MUST be validated against constraints. The system SHALL enforce:

- `max_attempts`: Integer between 1 and 10
- `backoff_strategy`: Enum value (exponential, linear, none)
- `backoff_initial_seconds`: Float between 0.1 and 10.0
- `backoff_max_seconds`: Float between 1.0 and 300.0
- `timeout_seconds`: Integer between 1 and 600
- `circuit_breaker.failure_threshold`: Integer between 3 and 10
- `rate_limit_per_second`: Optional float between 0.1 and 100.0

Invalid policy configurations MUST raise `ValidationError` with descriptive error messages.

#### Scenario: Valid resilience policy loaded

- **GIVEN** a policy YAML with `max_attempts: 3`, `backoff_strategy: exponential`, `timeout_seconds: 30`
- **WHEN** loading the policy config
- **THEN** a `ResiliencePolicyConfig` instance is returned
- **AND** all fields are validated against constraints

#### Scenario: Invalid backoff strategy rejected

- **GIVEN** a policy YAML with `backoff_strategy: "fibonacci"` (invalid enum value)
- **WHEN** loading the policy config
- **THEN** a `ValidationError` is raised with message "Invalid backoff_strategy. Must be one of: exponential, linear, none"

---

### Requirement: DAG Cycle Detection

Pipeline topology validation MUST detect cyclic dependencies and reject invalid configurations. The validation algorithm SHALL:

- Build a dependency graph from stage definitions
- Perform topological sort or depth-first search to detect cycles
- Report the specific stages involved in any detected cycle
- Ensure all stage dependencies reference existing stages (no dangling references)

#### Scenario: Acyclic pipeline accepted

- **GIVEN** a pipeline with stages A, B, C where A→B, B→C (linear chain)
- **WHEN** validating the topology
- **THEN** no cycles are detected
- **AND** the pipeline is accepted

#### Scenario: Cyclic pipeline rejected

- **GIVEN** a pipeline with stages A, B, C where A→B, B→C, C→A (cycle)
- **WHEN** validating the topology
- **THEN** a `ValidationError` is raised with message "Cycle detected in pipeline: A → B → C → A"

---

## ADDED Requirements

### Requirement: Configuration Hot Reload

The orchestration system SHOULD support hot reloading of pipeline topology and resilience policies without service restart. The system SHALL:

- Monitor `config/orchestration/` directory for file changes (using watchdog or inotify)
- Reload changed YAML files and revalidate configurations
- Update cached configs atomically (no partial updates)
- Emit CloudEvent `config.reloaded` when configs are successfully reloaded
- Log warnings for invalid configs without affecting running jobs

#### Scenario: Pipeline topology hot reloaded

- **GIVEN** Dagster daemon is running with `auto.yaml` loaded
- **WHEN** `auto.yaml` is modified to add a new stage
- **THEN** the config watcher detects the change within 5 seconds
- **AND** the new topology is loaded and validated
- **AND** subsequent job submissions use the updated pipeline
- **AND** in-flight jobs continue with the old topology

#### Scenario: Invalid config hot reload rejected

- **GIVEN** Dagster daemon is running
- **WHEN** `auto.yaml` is modified with invalid YAML syntax
- **THEN** the config reload fails with logged error
- **AND** the previous valid config remains in effect
- **AND** new job submissions continue using the old topology

---

## MODIFIED Requirements

### Requirement: Settings Management

The existing `AppSettings` class MUST be extended to include Dagster-specific configuration. New fields:

- `dagster_enabled: bool` - Feature flag for Dagster orchestration (default: False)
- `dagster_postgres_url: SecretStr` - PostgreSQL connection string for Dagster storage
- `dagster_webserver_host: str` - Dagster UI host (default: "0.0.0.0")
- `dagster_webserver_port: int` - Dagster UI port (default: 3000)
- `dagster_daemon_heartbeat_tolerance: int` - Seconds before daemon considered unhealthy (default: 60)
- `pipeline_config_dir: Path` - Directory for pipeline topology YAMLs (default: "config/orchestration/pipelines")
- `resilience_policy_file: Path` - Path to resilience policies YAML (default: "config/orchestration/resilience.yaml")

#### Scenario: Dagster settings loaded from environment

- **GIVEN** environment variables:
  - `MK_DAGSTER_ENABLED=true`
  - `MK_DAGSTER_POSTGRES_URL=postgresql://user:pass@localhost/dagster`
- **WHEN** `get_settings()` is called
- **THEN** settings.dagster_enabled is True
- **AND** settings.dagster_postgres_url.get_secret_value() returns the connection string
