# ADR 0007: Direction incremental updates trigger on index completion, with a split consent gate

Status: Accepted (2026-08-07 Helm P4 follow-up design)

Related: ADR 0004 (personal-library-guided discovery); ADR 0006 (unified search entry).

## Context

Direction incremental update (P3) keeps confirmed research directions current as new library papers arrive: placement assigns new papers to existing directions or the buffer pool; when the buffer pool crosses its trigger threshold, local reclustering plus an LLM diff produces suggestions (attach / new / split / merge) that enter a review queue. It currently runs only through the manual command `check-incremental-direction-updates`.

The product identity (personal research companion; knowledge evolution) argues that directions should follow new evidence without the researcher remembering to run a command. But two constraints shape the trigger design:

- **Machine suggestions never override user decisions** — suggestions only enter the review queue; nothing is auto-applied.
- **LLM diff generation requires model-processing consent** (the `personal-library-direction-generation` consent gate) and costs money. Placement, however, is pure local embedding similarity — no LLM, no consent needed.

Additionally, with automatic triggering the suggestions document is rewritten more often; its current whole-document replace semantics (new evidence supersedes old pending suggestions) would drop un-reviewed suggestions without the user noticing.

## Decision

### 1. Trigger on index completion

When `index-personal-library-fulltext` finishes with `indexed > 0` (new or changed papers), the incremental update runs automatically. Re-runs that only reuse existing embeddings do not trigger it.

### 2. Split the consent gate

- **Placement always runs** — local embedding similarity, consent-free; directions and the buffer pool stay current for every indexed library.
- **Recluster + LLM diff suggestions run only with model-processing consent**; without it the LLM part is skipped and recorded as pending authorization (待授权), surfaced in the review UI.

### 3. Suggestions never auto-apply

Auto-triggering only generates and queues suggestions; applying remains an explicit user review action (the existing apply/dismiss/lock flow).

### 4. Keep whole-document replace, add a supersede hint

The suggestions document keeps its whole-document replace semantics — it is the current-evidence snapshot. When a run supersedes un-reviewed suggestions, the user gets a status-bar / Notice hint that pending suggestions were updated, so nothing is silently dropped.

## Consequences

- Directions stay current without user action; LLM spend occurs only when consent is present and the buffer pool actually crosses its threshold.
- The consent-gate split changes the current `runIncrementalDirectionUpdate` entry (today the whole run is gated); the review UI gains a pending-authorization state for the LLM part.
- The review queue shows the latest suggestion snapshot; the supersede hint keeps replacement visible.
- Implementation is a new helm phase (P5) — search entry (ADR 0006) and this trigger design share the same phase planning.
