# Chunking Environment Setup

## Python Extras

Install framework adapters and advanced chunkers via the optional Poetry extra:

```bash
poetry install -E chunking
```

## NLTK Punkt Tokeniser

Download the Punkt sentence model once per environment:

```bash
python -m nltk.downloader punkt
```

## spaCy English Model

Install the small English model for spaCy-based sentence splitting:

```bash
python -m spacy download en_core_web_sm
```

## Validation

Run the quick health check to verify resources:

```bash
python - <<'PY'
from Medical_KG_rev.chunking.sentence_splitters import NLTKSentenceSplitter, SpacySentenceSplitter
print(NLTKSentenceSplitter().split("Sentence one. Sentence two."))
print(len(SpacySentenceSplitter().split("Sentence one. Sentence two.")))
PY
```
