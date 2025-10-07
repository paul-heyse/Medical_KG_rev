# Chunking Evaluation Harness

The evaluation harness in `eval/chunking_eval.py` computes segmentation and retrieval metrics for
any registered chunker.

## Running the Harness

```bash
python -m eval.chunking_eval
```

The CLI prints boundary F1, Recall@20, nDCG@10, and average latency for each configured chunker. To
customise the evaluated chunkers from code:

```python
from eval.chunking_eval import ChunkingEvaluationRunner

runner = ChunkingEvaluationRunner(["semantic_splitter", "llm_chaptering"])
results = runner.run()
```

## Metrics

* **Boundary F1** – compares chunk start offsets against gold annotations.
* **Recall@20** – proportion of relevant chunks retrieved in the top 20 lexical matches.
* **nDCG@10 / nDCG@20** – ranking quality for the top 10 and 20 chunks.
* **Latency** – average wall-clock latency per document in milliseconds.

Gold annotations live under `eval/gold/` and currently contain 10 documents for PMC, SPL, and
ClinicalTrials.gov samples.
