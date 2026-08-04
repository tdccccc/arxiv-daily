# P4 — union discovery and visible provenance

goal_ref: ../goal.md
updated: 2026-08-04

## Outcome

Every plugin daily run uses the deterministic union of configured manual topics and currently authorized, Core-eligible confirmed library directions without duplicating papers; the committed daily report and rebuilt Dashboard show strict occurrence-level discovery provenance that names manual topics, matched directions, representative prior papers, and metadata-and-abstract evidence depth, while manual-only behavior and existing commit, checkpoint, scheduler, summary, detail, and email semantics remain compatible.

## Assumptions

- P4 consumes only `evaluatePersonalLibraryInterestEligibility(...).eligibleDirections`. Generated proposals, disabled or merged directions, stale representative evidence, invalid documents, incompatible identities, and catalog/profile load failures never become discovery inputs.
- Sending confirmed direction text, cues, or representative metadata to the configured model endpoint is model processing and therefore requires current scope/depth/endpoint-bound authorization. Revocation or stale authorization degrades daily runs to manual-only discovery.
- Discovery provenance is a trusted selection fact, not generated novelty evidence. P5 will add validated abstract-level difference types and explanations without changing P4's authority boundary.
- Legacy manual filtering remains the exact path when no authorized eligible direction exists. Personalized filtering may use additional bounded model calls, but union, deduplication, ordering, and trusted provenance assembly remain deterministic Core behavior.
- Manual and both-source papers remain grouped under their selected manual topic. Library-only papers use one deterministic localized section after manual-topic sections; synthetic topics are not added to product settings.
- Committed daily Markdown is the durable occurrence authority. Paper Index and Dashboard state must be recoverable from strict bounded markers in that Markdown after interrupted derived-state updates.
- P4 changes daily Markdown and Dashboard projections only. Detailed-report generation, summary prompts/checkpoints, and email digest rendering remain unchanged until their later phase or explicit scope decision.

## Approach

Introduce bounded host-neutral discovery-input and occurrence-provenance contracts in Core. Preserve the legacy manual request/checkpoint path byte-for-byte when personalization is unavailable; otherwise classify eligible confirmed directions in deterministic bounded batches, strictly validate only paper and direction identities, and union those decisions with the existing manual result. Carry trusted source metadata through the existing paper pipeline, render visible and machine-readable provenance at the Markdown commit boundary, persist it per report occurrence in the Paper Index, and restore it during existing-daily repair and Dashboard history sync. The plugin constructs one coherent authorized eligibility snapshot for each daily pipeline and fails closed to manual-only during connection, output, endpoint, authorization, or document transitions.

## Tasks

- [ ] Define and adversarially test strict bounded Core discovery input, personalized classification, deterministic union/deduplication/order, trusted provenance, cancellation, and exact checkpoint compatibility while preserving the unchanged legacy manual path.
- [ ] Carry provenance through filtered papers and deterministic structured/fallback daily rendering, with a localized library-only section and a strict bounded machine-readable marker that cannot be forged by hostile metadata.
- [ ] Persist provenance per committed report occurrence in a backward-compatible Paper Index evolution; restore it through existing-daily repair and Dashboard history sync without erasing summaries, user state, paths, or repeated-date history.
- [ ] Build one plugin-owned immutable daily discovery snapshot from current authorization plus Core eligibility, wire every scheduler/manual daily entry through it, and fail closed across root/output/endpoint/revocation/reload races without changing manual single-paper fetch.
- [ ] Project durable occurrence provenance as separate Dashboard metadata, visibly naming manual/library/both sources, directions, representatives, and metadata-and-abstract depth without conflating query-time match reasons or P5 novelty.
- [ ] Prove manual-only compatibility, complete union provenance, authorization and privacy boundaries, deterministic commit/checkpoint recovery, and no summary/detail/email regression; run full repository quality suites and independent correctness/security reviews.
- [ ] Complete every accepted implementation chunk's independent technical-report handoff, mark P4 done while keeping P5 pending and the goal active, and create staged local commits without pushing.

## Verification

- Core filter tests prove direction-only, manual-only, and both-source selection; multiple-direction retention; canonical one-paper deduplication; stable ordering; complete deterministic batching; strict unknown/duplicate/malformed output rejection; cancellation; and no partial promotion.
- Checkpoint tests prove the legacy manual snapshot remains reusable only on the unchanged path, while personalized direction text/cues/representatives/order and prompt/result versions invalidate exact reuse; unrelated catalog entries and file-path-only changes do not.
- Rendering/parser/index tests prove Chinese and English provenance is identical for structured and fallback entries, hostile text cannot forge markers or raw HTML, old Markdown and Paper Index schemas remain loadable, and committed occurrence provenance survives restart, repair, repeated reports, and history rebuild.
- Plugin lifecycle tests prove only current authorized eligible directions enter daily calls; proposals and invalid/ineligible states degrade to manual-only; root/output/endpoint/revocation transitions cannot leak or promote stale snapshots; and scheduled, startup, date, force, pending, and retry runs get a fresh coherent snapshot.
- Dashboard tests prove provenance is visible without search, both-source and representative metadata are text-only, legacy rows remain compatible, and query-time `matchReasons` remain separate.
- Instrumented regressions prove P4 does not send roots, paths, PDF bytes, unrelated catalog records, authorization state, or credentials; does not alter summary prompts/checkpoints, detailed-report generation, email projection, manual fetch, daily commit authority, scheduler result semantics, or external-library write scope.
- Complete Core and plugin suites with the established Core heap allowance; node-runtime and CLI regressions; workspace typecheck/build; plugin production build; ESLint; boundary check; `git diff --check`; and independent correctness/security reviews pass.

## Abort / reshape triggers

- If complete direction classification cannot stay within explicit paper/direction/prompt/output bounds without silently omitting eligible directions or papers, stop and L2 reshape the filtering contract rather than introducing embeddings, RAG, or partial union semantics.
- If one personalized model response cannot be strictly validated using only supplied paper and direction identities, retain manual-only behavior and narrow the contract rather than accepting repaired, inferred, or model-authored provenance.
- If root/output/authorization transitions cannot provide a coherent daily snapshot without broad cancellation that regresses scheduler reliability, stop and L2 define a run-owned snapshot/provider boundary before continuing plugin wiring.
- If occurrence provenance cannot be recovered from committed Markdown without unsafe prose inference or an unbounded marker, stop and L2 define a smaller versioned marker grammar; do not make the mutable Paper Index the sole authority.
- If P4 begins generating novelty claims, changing detailed-report or email prose, adding reading workflows, or weakening metadata-versus-full-text evidence disclosure, move that work to P5 or a later initiative instead of widening this phase.
