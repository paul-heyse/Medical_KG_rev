# OpenSpec Instructions

Instructions for AI coding assistants using OpenSpec for spec-driven development.

## TL;DR Quick Checklist

- Search existing work: `openspec spec list --long`, `openspec list` (use `rg` only for full-text search)
- Decide scope: new capability vs modify existing capability
- Pick a unique `change-id`: kebab-case, verb-led (`add-`, `update-`, `remove-`, `refactor-`)
- Scaffold: `proposal.md`, `tasks.md`, `design.md` (only if needed), and delta specs per affected capability
- Write deltas: use `## ADDED|MODIFIED|REMOVED|RENAMED Requirements`; include at least one `#### Scenario:` per requirement
- Validate: `openspec validate [change-id] --strict` and fix issues
- Request approval: Do not start implementation until proposal is approved

## Three-Stage Workflow

### Stage 1: Creating Changes

Create proposal when you need to:

- Add features or functionality
- Make breaking changes (API, schema)
- Change architecture or patterns
- Optimize performance (changes behavior)
- Update security patterns

Triggers (examples):

- "Help me create a change proposal"
- "Help me plan a change"
- "Help me create a proposal"
- "I want to create a spec proposal"
- "I want to create a spec"

Loose matching guidance:

- Contains one of: `proposal`, `change`, `spec`
- With one of: `create`, `plan`, `make`, `start`, `help`

Skip proposal for:

- Bug fixes (restore intended behavior)
- Typos, formatting, comments
- Dependency updates (non-breaking)
- Configuration changes
- Tests for existing behavior

**Workflow**

1. Review `openspec/project.md`, `openspec list`, and `openspec list --specs` to understand current context.
2. Choose a unique verb-led `change-id` and scaffold `proposal.md`, `tasks.md`, optional `design.md`, and spec deltas under `openspec/changes/<id>/`.
3. Draft spec deltas using `## ADDED|MODIFIED|REMOVED Requirements` with at least one `#### Scenario:` per requirement.
4. Run `openspec validate <id> --strict` and resolve any issues before sharing the proposal.

### Stage 2: Implementing Changes

1. **Read proposal.md** - Understand what's being built
2. **Read design.md** (if exists) - Review technical decisions
3. **Read tasks.md** - Get implementation checklist
4. **Implement tasks sequentially** - Complete in order
5. **Mark complete immediately** - Update `- [x]` after each task
6. **Approval gate** - Do not start implementation until the proposal is reviewed and approved

### Stage 3: Archiving Changes

After deployment, create separate PR to:

- Move `changes/[name]/` → `changes/archive/YYYY-MM-DD-[name]/`
- Update `specs/` if capabilities changed
- Use `openspec archive [change] --skip-specs` for tooling-only changes
- Run `openspec validate --strict` to confirm the archived change passes checks

## Before Any Task

**Context Checklist:**

- [ ] Read relevant specs in `specs/[capability]/spec.md`
- [ ] Check pending changes in `changes/` for conflicts
- [ ] Read `openspec/project.md` for conventions
- [ ] Run `openspec list` to see active changes
- [ ] Run `openspec list --specs` to see existing capabilities

**Before Creating Specs:**

- Always check if capability already exists
- Prefer modifying existing specs over creating duplicates
- Use `openspec show [spec]` to review current state
- If request is ambiguous, ask 1–2 clarifying questions before scaffolding

### Search Guidance

- Enumerate specs: `openspec spec list --long` (or `--json` for scripts)
- Enumerate changes: `openspec list` (or `openspec change list --json` - deprecated but available)
- Show details:
  - Spec: `openspec show <spec-id> --type spec` (use `--json` for filters)
  - Change: `openspec show <change-id> --json --deltas-only`
- Full-text search (use ripgrep): `rg -n "Requirement:|Scenario:" openspec/specs`

## Quick Start

### CLI Commands

```bash
# Essential commands
openspec list                  # List active changes
openspec list --specs          # List specifications
openspec show [item]           # Display change or spec
openspec diff [change]         # Show spec differences
openspec validate [item]       # Validate changes or specs
openspec archive [change]      # Archive after deployment

# Project management
openspec init [path]           # Initialize OpenSpec
openspec update [path]         # Update instruction files

# Interactive mode
openspec show                  # Prompts for selection
openspec validate              # Bulk validation mode

# Debugging
openspec show [change] --json --deltas-only
openspec validate [change] --strict
```

### Command Flags

- `--json` - Machine-readable output
- `--type change|spec` - Disambiguate items
- `--strict` - Comprehensive validation
- `--no-interactive` - Disable prompts
- `--skip-specs` - Archive without spec updates

## Directory Structure

```
openspec/
├── project.md              # Project conventions
├── specs/                  # Current truth - what IS built
│   └── [capability]/       # Single focused capability
│       ├── spec.md         # Requirements and scenarios
│       └── design.md       # Technical patterns
├── changes/                # Proposals - what SHOULD change
│   ├── [change-name]/
│   │   ├── proposal.md     # Why, what, impact
│   │   ├── tasks.md        # Implementation checklist
│   │   ├── design.md       # Technical decisions (optional; see criteria)
│   │   └── specs/          # Delta changes
│   │       └── [capability]/
│   │           └── spec.md # ADDED/MODIFIED/REMOVED
│   └── archive/            # Completed changes
```

## Creating Change Proposals

### Decision Tree

```
New request?
├─ Bug fix restoring spec behavior? → Fix directly
├─ Typo/format/comment? → Fix directly
├─ New feature/capability? → Create proposal
├─ Breaking change? → Create proposal
├─ Architecture change? → Create proposal
└─ Unclear? → Create proposal (safer)
```

### Proposal Structure

1. **Create directory:** `changes/[change-id]/` (kebab-case, verb-led, unique)

2. **Write proposal.md:**

```markdown
## Why
[1-2 sentences on problem/opportunity]

## What Changes
- [Bullet list of changes]
- [Mark breaking changes with **BREAKING**]

## Impact
- Affected specs: [list capabilities]
- Affected code: [key files/systems]
```

3. **Create spec deltas:** `specs/[capability]/spec.md`

```markdown
## ADDED Requirements
### Requirement: New Feature
The system SHALL provide...

#### Scenario: Success case
- **WHEN** user performs action
- **THEN** expected result

## MODIFIED Requirements
### Requirement: Existing Feature
[Complete modified requirement]

## REMOVED Requirements
### Requirement: Old Feature
**Reason**: [Why removing]
**Migration**: [How to handle]
```

If multiple capabilities are affected, create multiple delta files under `changes/[change-id]/specs/<capability>/spec.md`—one per capability.

4. **Create tasks.md:**

```markdown
## 1. Implementation
- [ ] 1.1 Create database schema
- [ ] 1.2 Implement API endpoint
- [ ] 1.3 Add frontend component
- [ ] 1.4 Write tests
```

5. **Create design.md when needed:**
Create `design.md` if any of the following apply; otherwise omit it:

- Cross-cutting change (multiple services/modules) or a new architectural pattern
- New external dependency or significant data model changes
- Security, performance, or migration complexity
- Ambiguity that benefits from technical decisions before coding

Minimal `design.md` skeleton:

```markdown
## Context
[Background, constraints, stakeholders]

## Goals / Non-Goals
- Goals: [...]
- Non-Goals: [...]

## Decisions
- Decision: [What and why]
- Alternatives considered: [Options + rationale]

## Risks / Trade-offs
- [Risk] → Mitigation

## Migration Plan
[Steps, rollback]

## Open Questions
- [...]
```

## Spec File Format

### Critical: Scenario Formatting

**CORRECT** (use #### headers):

```markdown
#### Scenario: User login success
- **WHEN** valid credentials provided
- **THEN** return JWT token
```

**WRONG** (don't use bullets or bold):

```markdown
- **Scenario: User login**  ❌
**Scenario**: User login     ❌
### Scenario: User login      ❌
```

Every requirement MUST have at least one scenario.

### Requirement Wording

- Use SHALL/MUST for normative requirements (avoid should/may unless intentionally non-normative)

### Delta Operations

- `## ADDED Requirements` - New capabilities
- `## MODIFIED Requirements` - Changed behavior
- `## REMOVED Requirements` - Deprecated features
- `## RENAMED Requirements` - Name changes

Headers matched with `trim(header)` - whitespace ignored.

#### When to use ADDED vs MODIFIED

- ADDED: Introduces a new capability or sub-capability that can stand alone as a requirement. Prefer ADDED when the change is orthogonal (e.g., adding "Slash Command Configuration") rather than altering the semantics of an existing requirement.
- MODIFIED: Changes the behavior, scope, or acceptance criteria of an existing requirement. Always paste the full, updated requirement content (header + all scenarios). The archiver will replace the entire requirement with what you provide here; partial deltas will drop previous details.
- RENAMED: Use when only the name changes. If you also change behavior, use RENAMED (name) plus MODIFIED (content) referencing the new name.

Common pitfall: Using MODIFIED to add a new concern without including the previous text. This causes loss of detail at archive time. If you aren’t explicitly changing the existing requirement, add a new requirement under ADDED instead.

Authoring a MODIFIED requirement correctly:

1) Locate the existing requirement in `openspec/specs/<capability>/spec.md`.
2) Copy the entire requirement block (from `### Requirement: ...` through its scenarios).
3) Paste it under `## MODIFIED Requirements` and edit to reflect the new behavior.
4) Ensure the header text matches exactly (whitespace-insensitive) and keep at least one `#### Scenario:`.

Example for RENAMED:

```markdown
## RENAMED Requirements
- FROM: `### Requirement: Login`
- TO: `### Requirement: User Authentication`
```

## Troubleshooting

### Common Errors

**"Change must have at least one delta"**

- Check `changes/[name]/specs/` exists with .md files
- Verify files have operation prefixes (## ADDED Requirements)

**"Requirement must have at least one scenario"**

- Check scenarios use `#### Scenario:` format (4 hashtags)
- Don't use bullet points or bold for scenario headers

**Silent scenario parsing failures**

- Exact format required: `#### Scenario: Name`
- Debug with: `openspec show [change] --json --deltas-only`

### Validation Tips

```bash
# Always use strict mode for comprehensive checks
openspec validate [change] --strict

# Debug delta parsing
openspec show [change] --json | jq '.deltas'

# Check specific requirement
openspec show [spec] --json -r 1
```

## Happy Path Script

```bash
# 1) Explore current state
openspec spec list --long
openspec list
# Optional full-text search:
# rg -n "Requirement:|Scenario:" openspec/specs
# rg -n "^#|Requirement:" openspec/changes

# 2) Choose change id and scaffold
CHANGE=add-two-factor-auth
mkdir -p openspec/changes/$CHANGE/{specs/auth}
printf "## Why\n...\n\n## What Changes\n- ...\n\n## Impact\n- ...\n" > openspec/changes/$CHANGE/proposal.md
printf "## 1. Implementation\n- [ ] 1.1 ...\n" > openspec/changes/$CHANGE/tasks.md

# 3) Add deltas (example)
cat > openspec/changes/$CHANGE/specs/auth/spec.md << 'EOF'
## ADDED Requirements
### Requirement: Two-Factor Authentication
Users MUST provide a second factor during login.

#### Scenario: OTP required
- **WHEN** valid credentials are provided
- **THEN** an OTP challenge is required
EOF

# 4) Validate
openspec validate $CHANGE --strict
```

## Multi-Capability Example

```
openspec/changes/add-2fa-notify/
├── proposal.md
├── tasks.md
└── specs/
    ├── auth/
    │   └── spec.md   # ADDED: Two-Factor Authentication
    └── notifications/
        └── spec.md   # ADDED: OTP email notification
```

auth/spec.md

```markdown
## ADDED Requirements
### Requirement: Two-Factor Authentication
...
```

notifications/spec.md

```markdown
## ADDED Requirements
### Requirement: OTP Email Notification
...
```

## Best Practices

### Simplicity First

- Default to <100 lines of new code
- Single-file implementations until proven insufficient
- Avoid frameworks without clear justification
- Choose boring, proven patterns

### Complexity Triggers

Only add complexity with:

- Performance data showing current solution too slow
- Concrete scale requirements (>1000 users, >100MB data)
- Multiple proven use cases requiring abstraction

### Clear References

- Use `file.ts:42` format for code locations
- Reference specs as `specs/auth/spec.md`
- Link related changes and PRs

### Capability Naming

- Use verb-noun: `user-auth`, `payment-capture`
- Single purpose per capability
- 10-minute understandability rule
- Split if description needs "AND"

### Change ID Naming

- Use kebab-case, short and descriptive: `add-two-factor-auth`
- Prefer verb-led prefixes: `add-`, `update-`, `remove-`, `refactor-`
- Ensure uniqueness; if taken, append `-2`, `-3`, etc.

## Tool Selection Guide

| Task | Tool | Why |
|------|------|-----|
| Find files by pattern | Glob | Fast pattern matching |
| Search code content | Grep | Optimized regex search |
| Read specific files | Read | Direct file access |
| Explore unknown scope | Task | Multi-step investigation |

## Error Recovery

### Change Conflicts

1. Run `openspec list` to see active changes
2. Check for overlapping specs
3. Coordinate with change owners
4. Consider combining proposals

### Validation Failures

1. Run with `--strict` flag
2. Check JSON output for details
3. Verify spec file format
4. Ensure scenarios properly formatted

### Missing Context

1. Read project.md first
2. Check related specs
3. Review recent archives
4. Ask for clarification

## Quick Reference

### Stage Indicators

- `changes/` - Proposed, not yet built
- `specs/` - Built and deployed
- `archive/` - Completed changes

### File Purposes

- `proposal.md` - Why and what
- `tasks.md` - Implementation steps
- `design.md` - Technical decisions
- `spec.md` - Requirements and behavior

### CLI Essentials

```bash
openspec list              # What's in progress?
openspec show [item]       # View details
openspec diff [change]     # What's changing?
openspec validate --strict # Is it correct?
openspec archive [change]  # Mark complete
```

Remember: Specs are truth. Changes are proposals. Keep them in sync.

---

## Project-Specific Considerations: Medical_KG_rev

### System Architecture Context

This project implements a **multi-protocol API gateway and orchestration system** for biomedical knowledge integration. When working on this codebase, be aware of these architectural patterns:

#### Multi-Protocol Façade

- Single backend exposed through 5 protocols: REST (OpenAPI + JSON:API + OData), GraphQL, gRPC, SOAP, AsyncAPI/SSE
- Protocol handlers share common service layer - no duplicate business logic
- Always implement protocol-agnostic logic first, then add protocol-specific wrappers

#### Federated Data Model

- **Core entities**: Document, Block, Section, Entity, Claim, Organization
- **Domain overlays**: Medical (FHIR-aligned), Financial (XBRL), Legal (LegalDocML)
- Use discriminated unions for domain-specific extensions
- All models use Pydantic v2 with strict validation

#### Adapter SDK Pattern

- Data sources plug in via BaseAdapter interface: fetch() → parse() → validate() → write()
- Simple REST APIs: Define in YAML (Singer/Airbyte-inspired)
- Complex sources: Implement Python adapter class
- Each adapter manages its own rate limits and retry logic

#### Two-Phase Pipeline

- **Auto-pipeline**: Fast sources (metadata → chunk → embed → index)
- **Manual pipeline**: GPU-bound (metadata → PDF fetch → MinerU → postpdf → chunk → embed → index)
- Ledger tracks document processing stage for idempotency

#### Fail-Fast Philosophy

- GPU operations: Fail immediately if GPU unavailable (no CPU fallback)
- Validation: Reject at entry points (don't propagate bad data)
- External APIs: Validate IDs before making requests
- Contracts: Schemathesis/GraphQL Inspector/Buf prevent spec drift

### Implementation Guidelines

#### When Adding New Data Sources

1. Check if RESTAdapter + YAML config sufficient
2. If complex (SOAP, PDF, special auth): Implement Python adapter
3. Add to adapter registry with source name
4. Define rate limits in config
5. Add comprehensive tests with mocked responses
6. Update OpenAPI endpoints if user-facing

#### When Adding New Extraction Types

1. Extend `/extract/{kind}` endpoint (not new endpoint)
2. Define extraction template (PICO, effects, AE, dose, eligibility)
3. Implement span-grounding validation
4. Ensure provenance tracking (ExtractionActivity node)
5. Add SHACL shapes for graph validation if new node types

#### When Working with GPU Services

1. Always check GPU availability on startup
2. Return clear error immediately if unavailable (fail-fast)
3. Implement batch processing for efficiency
4. Add Prometheus metrics for GPU utilization
5. Use gRPC for inter-service communication (not REST)

#### When Implementing Retrieval Features

1. Consider all three strategies: BM25 (full-text), SPLADE (sparse), Dense (vectors)
2. Use fusion ranking (RRF) to combine results
3. Test P95 latency < 500ms requirement
4. Implement span highlighting for results
5. Support OData filters on retrieval endpoints

#### Security Considerations

1. **Multi-tenancy**: Every query MUST filter by tenant_id from JWT
2. **Scopes**: Enforce at endpoint level (ingest:write, kg:read, etc.)
3. **Rate limiting**: Implement per-client and per-endpoint
4. **Audit logging**: Log all mutations with user, action, resource, timestamp
5. **Secrets**: Use Vault or env vars, never hardcode

#### Performance & Observability

1. Add OpenTelemetry spans for all significant operations
2. Emit Prometheus metrics with labels (endpoint, status, tenant)
3. Include correlation ID in all logs
4. Set SLO alerts (P95 latency, error rate, GPU saturation)
5. Profile GPU memory usage to prevent OOM

### Domain-Specific Patterns

#### Biomedical Data Validation

- **NCT IDs**: `NCT\d{8}` format
- **DOIs**: `10.\d{4,}/.*` pattern
- **PMCIDs**: `PMC\d+` format
- **RxCUIs**: Validate via RxNorm adapter
- **ICD-11 codes**: Validate via WHO API
- **Units**: Enforce UCUM standard for medical measurements

#### Provenance Tracking

Every extracted fact MUST link to:

- Source document (doc_id)
- Extraction method (ExtractionActivity with model_name, prompt_version)
- Original text span (start, end, text)
- Timestamp (UTC, ISO format)

#### FHIR Alignment

When adding medical domain features:

- Map to FHIR resources where possible (Evidence, ResearchStudy, MedicationStatement)
- Use FHIR terminology (CodeableConcept, Identifier, Reference)
- Maintain FHIR extension points for custom fields

### Testing Strategy

#### Contract Tests (Required)

- **REST**: Schemathesis generates tests from OpenAPI spec
- **GraphQL**: GraphQL Inspector detects breaking changes
- **gRPC**: Buf breaking change detection on .proto files
- Run in CI on every PR - failing contract tests block merge

#### Performance Tests (Required)

- k6 scripts with thresholds (P95 < 500ms for retrieval)
- Test concurrent job processing (5+ simultaneous ingests)
- GPU service load testing (batch sizes, memory limits)
- Run nightly or on release branches

#### Integration Tests (Required)

- Docker Compose test environment with all services
- Test multi-adapter chaining (OpenAlex → Unpaywall → MinerU)
- Test two-phase pipeline end-to-end
- Test multi-tenant isolation

### Common Pitfalls to Avoid

1. **Protocol Divergence**: Don't implement logic in REST handler that GraphQL can't access. Use shared service layer.

2. **Adapter Tight Coupling**: Don't hardcode external API URLs/schemas. Use adapter configs that can be updated without code changes.

3. **Tenant Leakage**: Always filter by tenant_id. Never trust client-provided tenant in query params - extract from JWT.

4. **CPU Fallback**: For GPU services, explicitly fail if GPU unavailable. Don't silently fall back to CPU.

5. **Partial Saves**: Validate completely before persisting. Failed validation should not leave partial data.

6. **Duplicate Provenance**: Use MERGE operations with idempotency keys in Neo4j to prevent duplicate nodes/edges.

7. **Unbounded Queries**: Always paginate. Add $top/$skip support to list endpoints.

8. **Missing Correlation IDs**: Generate and propagate correlation ID through all service calls for tracing.

### Useful Commands for This Project

```bash
# Start local development stack
docker-compose up -d

# Start API gateway
python -m Medical_KG_rev.gateway.main

# Start background workers
python -m Medical_KG_rev.orchestration.workers

# Run adapter with sample data (via API)
curl -X POST http://localhost:8000/v1/ingest/clinicaltrials \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"data": {"type": "ingestion", "attributes": {"nct_ids": ["NCT04267848"]}}}'

# Test retrieval performance
k6 run tests/performance/retrieve_latency.js

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Validate UCUM units
python -c "from Medical_KG_rev.validation import UCUMValidator; v = UCUMValidator(); print(v.validate('10 mg/dL'))"

# Validate FHIR resource
python -c "from Medical_KG_rev.validation.fhir import FHIRValidator; v = FHIRValidator(); v.validate_resource('Evidence', {...})"

# Validate graph shapes
python -c "from Medical_KG_rev.kg.shacl import ShaclValidator; v = ShaclValidator(); v.validate_graph(graph_data)"

# Generate OpenAPI spec
python scripts/generate_api_docs.py

# Export GraphQL schema
python scripts/update_graphql_schema.py

# Compile gRPC protos
buf generate

# Run contract tests
pytest tests/contract/

# Run performance tests
k6 run tests/performance/gateway_smoke_test.js

# Run all quality checks
pre-commit run --all-files

# Check for breaking changes in gRPC
bash scripts/run_buf_checks.sh

# MinerU smoke test
bash scripts/test_vllm_api.sh
```

### Resources & References

- **Engineering Blueprint**: `1) docs/Engineering Blueprint_ Multi-Protocol API Gateway & Orchestration System.pdf`
- **Biomedical APIs**: `1) docs/Section A_ Public Biomedical APIs for Integration.pdf`
- **Architecture**: `1) docs/System Architecture & Design Rationale.md`
- **Implementation Roadmap**: `IMPLEMENTATION_ROADMAP.md`
- **Project Context**: `openspec/project.md`
- **Active Changes**: Run `openspec list` to see current proposals

### Change Implementation Sequence

Follow this order for the 9 major changes:

1. `add-foundation-infrastructure` - Core models and utilities (48 tasks) ✅ IMPLEMENTED
2. `add-multi-protocol-gateway` - API façade layer (62 tasks) ✅ IMPLEMENTED
3. `add-biomedical-adapters` - Data source integrations (49 tasks) ✅ IMPLEMENTED
4. `add-ingestion-orchestration` - Kafka pipeline (36 tasks) ✅ IMPLEMENTED
5. `add-gpu-microservices` - ML/AI services (33 tasks) ✅ IMPLEMENTED
6. `add-knowledge-graph-retrieval` - Storage and search (43 tasks) ✅ IMPLEMENTED
7. `add-security-auth` - OAuth and multi-tenancy (49 tasks) ✅ IMPLEMENTED
8. `add-devops-observability` - CI/CD and monitoring (69 tasks) ✅ IMPLEMENTED
9. `add-domain-validation-caching` - UCUM, FHIR, HTTP caching (73 tasks) ✅ IMPLEMENTED

**Total**: 462 tasks across 16+ capabilities - ALL IMPLEMENTED

Each change builds on previous ones. The system is now production-ready.

### Getting Unstuck

**Q: How do I add a new biomedical data source?**
A: See "When Adding New Data Sources" section above. Start with adapter YAML, test with mock responses, then integrate with orchestrator.

**Q: My GPU service is running on CPU. Is that okay?**
A: No. GPU services must fail-fast if GPU unavailable. Check `CUDA_VISIBLE_DEVICES` and add explicit GPU checks.

**Q: How do I ensure my feature works across all protocols?**
A: Implement in service layer (protocol-agnostic), then add thin wrappers in REST/GraphQL/gRPC handlers. Test via all protocols.

**Q: Where does ontology mapping happen?**
A: In extraction/mapping phase after initial ingestion. RxNorm/ICD adapters are called by mapping workers, not directly by ingest endpoints.

**Q: How do I debug a failed job?**
A: Check ledger state, correlation ID in logs, trace in Jaeger, and dead letter queue for retries.

**Q: How do I validate UCUM units?**
A: Use `UCUMValidator` from `Medical_KG_rev.validation`. It uses `pint` library for unit validation. Example: `validator.validate("10 mg/dL")`.

**Q: How do I validate FHIR resources?**
A: Use `FHIRValidator` from `Medical_KG_rev.validation.fhir`. It validates against FHIR R5 schemas using jsonschema.

**Q: Where are the extraction templates defined?**
A: In `Medical_KG_rev.services.extraction.templates`. Templates include: PICO, EffectMeasure, AdverseEvent, DoseRegimen, EligibilityCriteria.

**Q: How do I enable HTTP caching?**
A: ETags and Cache-Control headers are automatically added by the REST API. Check `gateway/rest/router.py` for implementation.

**Q: How do I add a new SHACL shape?**
A: Add shape definitions to `src/Medical_KG_rev/kg/shapes.ttl` and update `ShaclValidator` in `kg/shacl.py`.
