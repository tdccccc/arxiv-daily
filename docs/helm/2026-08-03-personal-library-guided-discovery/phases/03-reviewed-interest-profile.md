# P3 — researcher-reviewed interest profile

goal_ref: ../goal.md
updated: 2026-08-03

## Outcome

From a loaded personal-library catalog and a currently authorized metadata-and-abstract model endpoint, the Obsidian plugin can generate bounded candidate research directions with small representative paper sets, let the researcher inspect, correct, merge, confirm, disable, and remove them, and durably reload the confirmed interest profile; only active confirmed directions that still validate against their representative catalog evidence are exposed as eligible library-derived discovery inputs, while P3 does not change daily filtering or reports.

## Assumptions

- The authoritative researcher record is strict structured Vault data under the configured index root, edited through typed Core mutations and plugin UI. Generated proposals are a separate replaceable document. Markdown authority or a generated Markdown projection is not required in P3.
- The paper-level catalog remains the source of metadata-and-abstract evidence. Proposal prompts use canonical paper keys and bounded bibliographic/abstract fields, never absolute roots, logical file paths, PDF bytes, unrelated files, or plugin-local authorization data.
- Valid library-processing consent is required only when catalog evidence is sent to the configured endpoint. Reviewing or editing already persisted proposals and confirmed directions remains local and does not require active model authorization.
- A generated proposal is never a discovery input. Confirmation is an explicit researcher action; editing or merging proposals does not implicitly confirm them. Disabled and merged confirmed records are retained for inspection but are ineligible.
- Confirmed directions have durable opaque identities, bounded discovery cues, and 1–5 unique representative `paperKey` values. Compatibility is assessed from the exact representative metadata-and-abstract evidence rather than broad catalog revision alone, so unrelated catalog changes do not deactivate a direction.
- P4 will consume the eligibility projection and combine it with manual topics. P3 may define and test that projection but must not modify `ArxivPipeline`, product topics, checkpoints, Markdown reports, Dashboard paper queries, scheduler behavior, or delivery.

## Approach

Keep three explicit authority layers in Core: the P2 catalog is a replaceable fact projection; a versioned proposal document is replaceable bounded model output tied to an exact canonical catalog-input and generation-contract fingerprint; and a separate versioned confirmed-profile document is the researcher-owned authority. Both P3 documents use strict schemas, atomic whole-document persistence, valid-backup recovery, fail-closed corruption handling, semantic revisions, and stale-write protection.

A bounded Core proposer deterministically selects and renders catalog evidence, partitions oversized catalogs by explicit paper/character budgets, validates every structured model result against supplied paper keys, and synthesizes a final proposal set without granting it authority. The plugin gates the first model request on current consent, owns cancellation and stale catalog/connection checks, and exposes one review experience shared by settings and commands. Pure Core review mutations perform candidate correction/merge/removal and confirmed-direction confirmation/edit/merge/disable/removal. A diagnostic eligibility projection admits only active confirmed directions whose representatives and evidence fingerprint still match the current catalog.

## Tasks

- [x] Define and adversarially test strict Core schemas, canonical fingerprints, bounds, opaque identity/lineage, stale diagnostics, and discovery-eligibility semantics for separate proposal and confirmed-profile documents.
- [x] Implement proposal and confirmed-profile stores under the configured Vault index root with atomic writes, valid-backup repair, fail-closed corruption handling, semantic no-op revisions, optimistic stale-write rejection, path-scoped serialization, and independent recovery.
- [x] Implement and test pure researcher-review transactions for candidate correction/merge/removal and explicit confirmation plus confirmed-direction edit/merge/disable/re-enable/removal, preserving authority boundaries and representative evidence integrity.
- [ ] Implement a bounded, cancellable Core direction proposer with deterministic evidence selection/batching, prompt-injection containment, strict reference validation and typed retry, 1–12 candidate directions, 1–5 representatives each, and no path/PDF/unrelated-content disclosure.
- [ ] Wire plugin generation/reload lifecycle with current-consent enforcement, a distinct operation kind, duplicate rejection, root/output/endpoint/revocation/unload cancellation, catalog-snapshot revalidation before commit, and no daily-workflow side effects.
- [ ] Add an inspectable Obsidian review surface reachable from Personal library settings and commands, separating Proposed from Confirmed state and requiring explicit actions for confirmation, merging, disabling, and removal; display representative paper identity, evidence depth, staleness, and missing-evidence diagnostics without exposing filesystem paths.
- [ ] Add restart, concurrency, authorization, stale-catalog, malformed-model, corruption, and no-impact regression coverage; run affected full suites, security review, technical-report handoffs for accepted chunks, workspace typecheck/build/lint/boundary checks, and a staged phase-completion commit.

## Verification

- Core schema/store tests prove exact-key and future-schema rejection, canonical fingerprints, prototype-safe identities, bounded fields and representatives, semantic no-op behavior, stale-write rejection, backup repair, corruption fail-closed behavior, and concurrent mutation serialization.
- Review tests prove generated or edited candidates remain ineligible until an explicit confirm transaction; proposal merges remain proposals; active confirmed edits remain researcher-authoritative; disabled and merged records are ineligible; missing or changed representative evidence fails closed without silently deleting the researcher record.
- Proposer tests prove deterministic batching and synthesis stay within configured paper/character limits, accept only supplied canonical paper keys, reject malformed or invented structured output after bounded typed retries, propagate cancellation, escape untrusted paper text, and omit absolute roots, logical paths, PDF bytes, authorization state, and unrelated content.
- Plugin tests prove generation cannot make an LLM call without current consent, local review remains available after revocation, stale connection/catalog/output operations cannot promote proposals, restart independently reloads proposals and confirmed profile, and one corrupt document does not silently erase or replace the other.
- UI tests prove Proposed and Confirmed states are visibly separate, corrections do not confirm, confirmation/merge/disable/remove are explicit, representative papers and metadata-and-abstract evidence are inspectable, stale or missing evidence is visible, and model text is rendered only as text.
- Instrumented regressions prove P3 does not mutate manual topics/categories, invoke daily runs or `ArxivPipeline`, change paper-index/checkpoint/delivery state, write daily Markdown, alter Dashboard paper queries, or send external-library PDF/file content beyond catalog metadata and abstracts to the authorized endpoint.
- Core and plugin full tests; node-runtime and CLI regression suites where affected; workspace typecheck/build; lint; boundary check; `git diff --check`; and independent security review pass.

## Abort / reshape triggers

- If representative real catalogs cannot fit bounded deterministic proposal generation without embeddings or full-text retrieval, stop and L2 reshape around a smaller explicit catalog-evidence selection step rather than introducing RAG or silently truncating the researcher’s library context.
- If generated output cannot be validated solely against supplied canonical paper keys and strict bounded contracts, retain the last valid proposals and narrow the contract instead of accepting repaired or partially trusted model output.
- If editable Markdown authority is required for acceptable researcher control, stop and L2 define one authoritative round-trip grammar and conflict policy; do not ship two independently writable authorities.
- If preserving a confirmed direction across catalog evolution requires silently replacing representative papers or weakening evidence checks, keep it visible but ineligible and require researcher repair.
- If P3 begins changing daily selection, provenance, novelty, reports, scheduler, or delivery behavior, move that work to P4/P5 instead of widening this phase.
