# Documentation map

Use this map to find the smallest document that answers a question.

## User-facing docs

- [`../README.md`](../README.md) — package overview, quick examples, validation model, and development checks.
- [`api/schema1.md`](api/schema1.md) — frozen schema-1 public API and graph contract.
- [`contributing/coarseners.md`](contributing/coarseners.md) — implementation guidance for new coarseners.

## Design notes

- [`design/star_coarsener.md`](design/star_coarsener.md) — historical simple star coarsener notes.
- [`design/parametric_star_coarsener.md`](design/parametric_star_coarsener.md) — parametric star coarsener semantics.
- [`design/edge_bpe_numba_experiment.md`](design/edge_bpe_numba_experiment.md) — optional Numba edge-BPE backend.
- [`design/greedy_bpe.md`](design/greedy_bpe.md) — greedy edge-BPE scheduling variant.

## Contract traceability

- [`audit/contract_traceability.md`](audit/contract_traceability.md) — contract-to-code/test traceability.
- [`audit/SCHEMA_CLARIFICATIONS.md`](audit/SCHEMA_CLARIFICATIONS.md) — implementation clarifications for the frozen schema.
- [`../tests/baselines/`](../tests/baselines/) — protected source snapshots and the frozen schema hash used by tests.


## Executable examples

The source-checkout notebooks live under [`../notebooks/`](../notebooks/), including the two-stage Star → BPE round-trip tutorial.
