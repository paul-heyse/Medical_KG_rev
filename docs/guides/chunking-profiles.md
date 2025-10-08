# Chunking Profiles

Profile-driven chunking standardizes how different biomedical sources are
processed. Each profile specifies the target token budget, section alignment,
sentence segmenter, and filter pipeline.

## Profile Overview

| Profile | Domain | Chunker | Sentence Segmenter | Key Metadata |
| --- | --- | --- | --- | --- |
| `pmc-imrad` | Literature | LangChain RecursiveCharacterTextSplitter with IMRaD-aware metadata hooks | Hugging Face tokenizer (abbreviation merging enabled) | `Abstract`, `Introduction`, `Methods`, `Results`, `Discussion`, with provenance offsets |
| `ctgov-registry` | Clinical trials | Custom registry chunker built on `SentenceWindowNodeParser` | syntok | Eligibility, Outcome, Adverse Event, Results sections |
| `spl-label` | Drug labels | SPL-aware recursive chunker with LOINC mapping | Hugging Face tokenizer | Indications, Dosage, Warnings, Adverse Reactions, Clinical Pharmacology |
| `guideline` | Clinical guidelines | Recommendation/evidence chunker with heuristic paragraph aggregation | syntok | Recommendation statements, evidence summaries, strength/grade metadata |

## Configuration Files

Configuration lives under `config/chunking/profiles/*.yaml`. Example excerpt
from `pmc-imrad`:

```yaml
name: pmc-imrad
domain: literature
target_tokens: 800
sentence_splitter: huggingface
chunker:
  type: langchain
  options:
    splitter: recursive
filters:
  - boilerplate
  - deduplicate-furniture
  - prune-references
```

To load profiles programmatically:

```python
from Medical_KG_rev.services.chunking.profiles.loader import load_profiles

profiles = load_profiles()
imrad = profiles["pmc-imrad"]
print(imrad.target_tokens)
```

## Extending Profiles

1. Create a new YAML file under `config/chunking/profiles/`.
2. Add the profile to the chunker registry via
   `Medical_KG_rev.services.chunking.registry.register_profile`.
3. Update documentation and run `pytest tests/chunking/test_profiles.py`.
4. Verify the profile-specific chunker passes `scripts/check_chunking_dependencies.py`.

## Operational Guidelines

- Keep token budgets under the embedding model limit with a 10% safety margin.
- Ensure every chunk emits `section_label`, `intent_hint`, and ordered
  `char_offsets`.
- Document fallback behavior (naive sentence splitting) in case of tokenizer
  failures and monitor via Prometheus metrics.

## Related Resources

- [Chunking & Parsing Runtime Guide](./chunking.md)
- [MinerU Two-Phase Gate Runbook](../runbooks/mineru-two-phase-gate.md)
