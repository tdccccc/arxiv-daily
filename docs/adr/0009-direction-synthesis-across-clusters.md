# ADR 0009: Direction candidates are synthesized across clusters before review

Status: Superseded for initial library proposals by ADR 0014 §1 (2026-09-05)

The initial proposal now organizes evidence groups directly into a few topics and directions in one model stage. Per-cluster extraction and subsequent synthesis are retired. The requirement not to silently lose evidence survives as complete, unique group assignment; consent and cancellation boundaries remain unchanged. The original decision below records why concatenating cluster-level directions was inadequate.

Related: ADR 0004 (personal-library-guided discovery); ADR 0007 (incremental trigger and split consent gate).

## Context

The clustered direction proposer groups library papers locally by embedding similarity, then runs **one extraction call per cluster** and concatenates every cluster's candidates into the proposal. Clusters never see each other, so two clusters covering the same research direction each name it independently and both names reach the review page.

The earlier unclustered proposer had a grouping-and-synthesis stage that merged equivalent themes across batches. The clustered path dropped that stage; nothing replaced it.

Measured on the test library (91 papers, two knowledge-base shards): 22 candidates, of which at least seven cross-shard pairs are the same direction under two names — for example `Outer-halo profiles as tests of gravity` and `Cluster mass profiles from interiors to turnaround scales and tests of gravity`, sharing the cues *alternative gravity*, *cluster outskirts*, *cluster kinematics*. By their discovery cues the 22 candidates collapse to roughly 6–8 distinct directions.

Fragmentation is not only a review-page annoyance. The daily personalized classifier chunks directions **12 per batch** and calls the model once per (direction chunk × paper batch). Nineteen confirmed directions means two direction chunks; eight means one. Duplicated directions therefore roughly **double the daily classification cost** while adding no coverage, and the same paper gets matched by two synonymous directions, duplicating its discovery-source markers in the report.

## Decision

### 1. Synthesize after clustering, before the proposal is persisted

After every cluster's extraction returns, one further model call reviews the combined candidate set and merges candidates that express the same research direction. The proposal the researcher reviews is the synthesized set, not the raw per-cluster concatenation.

This restores, for the clustered path, the stage the unclustered path already had. It is a one-time cost per proposal, paid to reduce a recurring daily cost.

### 2. Synthesis merges; it never drops

Synthesis may combine candidates and may mark them. It must not remove a candidate from the proposal. A direction the researcher never sees cannot be corrected, and an emerging interest with thin evidence is exactly the case where silent removal is most harmful.

### 3. Thin-evidence candidates are surfaced unchecked, not hidden

A candidate whose evidence is too thin to stand on its own — for example the single-paper `Quantum PCP and robust local verification of many-body states` in a library otherwise about cluster cosmology — is presented like any other candidate but marked, and bulk acceptance leaves it unselected by default. Including it stays a deliberate act; excluding it costs nothing.

### 4. Consent and cancellation are unchanged

Synthesis is part of direction generation and runs under the existing `personal-library-direction-generation` consent gate and cancellation scope. It introduces no new authorization surface and no new processing depth.

## Consequences

- Proposal generation costs one additional model call; daily classification cost roughly halves for a library whose directions would otherwise exceed one 12-direction chunk.
- The review page shows fewer, broader candidates. Bulk acceptance (the researcher's chosen default path) therefore produces a profile that is cheaper to run and free of synonymous directions.
- Synthesis is a model call and can merge two directions the researcher considers distinct. The merge is visible and correctable on the review page before confirmation, and confirmed directions can still be merged or split afterwards through the existing review actions.
- Candidate counts stop tracking cluster counts, so proposal size no longer leaks the sharding of the knowledge base.
