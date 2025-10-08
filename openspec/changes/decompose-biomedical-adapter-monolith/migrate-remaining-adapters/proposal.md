## Why

With the OpenAlex adapter now extracted into `adapters/openalex/`, the remaining biomedical adapters (Unpaywall, Crossref, PMC, CORE, ClinicalTrials, etc.) are still embedded inside the monolithic `adapters/biomedical.py`. Keeping them there blocks us from:

- Enforcing the new pluggy-based interface/mixin conventions across every biomedical source.
- Sharing DOI/open-access/PDF discovery logic in one place, which will be needed for downstream PDF persistence and MinerU gating.
- Running adapter-scoped unit tests that use focused fixtures instead of the legacy integration-style suites.

Migrating the rest of the adapters into dedicated packages ensures the whole biomedical surface area follows the same lifecycle contract that the OpenAlex rewrite established.

## What Changes

1. **Module decomposition**
   - Create `adapters/<source>/` packages for Unpaywall, Crossref, PMC (and their immediate siblings that still live in the monolith).
   - Port each class into its module, keeping public imports in `adapters/__init__.py` for backward compatibility.

2. **Shared mixins + utilities**
   - Introduce mixins/utilities for common behaviours (HTTP wrapper, DOI normalisation, open-access metadata shaping, pagination) so adapters stop duplicating logic.
   - Adopt those mixins while migrating each adapter, mirroring the pattern used in the new OpenAlex implementation.

3. **Plugin + registry updates**
   - Update `adapters/plugins/domains/biomedical` to point at the new classes without breaking plugin metadata/capabilities.
   - Ensure each adapter module exposes a plugin-friendly metadata helper similar to OpenAlex.

4. **Test coverage**
   - Expand `tests/adapters/test_biomedical_adapters.py` (or split per module) with stubs that validate fetch→parse behaviour for every migrated adapter.
   - Add regression cases for PDF URLs / OA metadata parity, using the OpenAlex test harness as a template.

5. **Documentation + specs**
   - Refresh `openspec/changes/decompose-biomedical-adapter-monolith/tasks.md` checkboxes as modules move.
   - Capture mixin usage and module layout in `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md` so future adapter authors follow the new pattern.

## Impact

- **Codebase**: Removes the last large class cluster from `adapters/biomedical.py`; adds new packages under `src/Medical_KG_rev/adapters/`.
- **Specs**: Keeps the existing decomposition spec aligned; no new requirements beyond those already defined.
- **Tests**: Unit coverage shifts from the monolithic test file to adapter-specific stubs; CI suite remains the same.
- **Risks**: Plugin registration order and imports must remain intact—introduce wrappers if needed until the gateway service is ready for full hot-swappable discovery.

## Success Criteria

- No biomedical adapter logic remains in the legacy monolith after migration.
- Every migrated adapter exports metadata, PDF URLs, and identifier fields consistent with the OpenAlex contract.
- Adapter plugins load without manual wiring changes, and the existing ingestion pipelines run with the new modules.

## Tasks

1. **Scaffold packages**: create `src/Medical_KG_rev/adapters/{unpaywall,crossref,pmc,core,...}/__init__.py` and module skeletons; wire temporary re-exports so imports stay stable.
2. **Extract adapter logic**: move each adapter class out of `adapters/biomedical.py`, adapting to mixin utilities and ensuring docstrings/metadata match the new layout.
3. **Introduce shared mixins**: implement HTTP, DOI/open-access, pagination, and metadata shaping helpers; refactor migrated adapters to consume them.
4. **Update plugins**: repoint `adapters/plugins/domains/biomedical` registrations to the new classes, preserving capability metadata and health checks.
5. **Expand tests**: add adapter-specific fixtures mirroring the OpenAlex stub harness, validating fetch→parse behaviour and PDF/OA metadata for each adapter.
6. **Documentation + spec sync**: refresh `openspec/.../tasks.md` checkboxes, update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`, and note migration guidelines.
7. **Regression verification**: run targeted ingestion flows (e.g., Unpaywall + PMC) to confirm pipelines and stage plugins operate with the decomposed modules.
