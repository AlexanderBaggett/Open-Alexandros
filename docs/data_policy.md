# Data Policy

Alexandros does not curate or redistribute a large-scale training corpus in v1.
The repository only includes tiny synthetic fixtures for CI and mechanism tests.

## Dataset Ownership

Downstream trainers are responsible for:

- selecting datasets and tokenizers;
- verifying dataset licenses, terms, and allowed use;
- documenting provenance, versions, and checksums;
- filtering, deduplication, and train/validation/test splits;
- publishing dataset cards when releasing trained checkpoints.

The repository provides `docs/templates/dataset_card.md` as the minimum card
shape for downstream runs.

## License And Copyright Tracking

Any published trained checkpoint should state:

- dataset names, versions, and source URLs or citations;
- license and terms for each dataset;
- whether commercial use, redistribution, and derivative models are allowed;
- whether generated teacher outputs were used and under which terms;
- any excluded or filtered sources.

Do not imply that Alexandros fixtures, configs, or smoke scripts grant rights to
third-party datasets.

## PII And Privacy Filtering

A real training run should define:

- PII detection and removal policy;
- handling of emails, phone numbers, addresses, secrets, and credentials;
- audit sampling process;
- opt-out or takedown path if the downstream release supports one;
- privacy leakage evaluation before publishing checkpoints.

## Contamination And Decontamination

Benchmarks should document:

- benchmark names and versions;
- matching/decontamination method;
- whether exact, near-duplicate, or paraphrase matches were removed;
- tokenizer and normalization used for matching;
- residual contamination risk.

## No Scraping In V1

The Alexandros repository must not add internet scraping, web-scale data
collection, or redistribution scripts unless project scope changes explicitly.
External trainers may build their own data pipelines, but those pipelines are
outside the v1 repository boundary.
