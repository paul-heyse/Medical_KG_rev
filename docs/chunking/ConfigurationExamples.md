# Chunking Configuration Examples

The chunking subsystem is configured through `config/chunking.yaml`. This document provides example
profiles for common corpora.

## Default Research Profile

```yaml
default_profile: research
profiles:
  research:
    enable_multi_granularity: true
    primary:
      strategy: semantic_splitter
      granularity: paragraph
      params:
        tau_coh: 0.82
        min_tokens: 200
    auxiliaries:
      - strategy: section_aware
        granularity: section
      - strategy: sliding_window
        granularity: window
        params:
          target_tokens: 400
          overlap_ratio: 0.2
```

## ClinicalTrials.gov Profile

```yaml
profiles:
  ctgov:
    enable_multi_granularity: true
    primary:
      strategy: clinical_role
      granularity: paragraph
    auxiliaries:
      - strategy: llm_chaptering
        granularity: section
        params:
          prompt_version: v2
      - strategy: table
        granularity: table
```

## Drug Label (SPL) Profile

```yaml
profiles:
  spl:
    enable_multi_granularity: false
    primary:
      strategy: layout_aware
      granularity: section
      params:
        overlap_threshold: 0.25
```
