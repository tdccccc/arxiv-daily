# Personal-library-guided daily discovery

status: active
updated: 2026-08-04
owner: current-session

## Intent

Connect a researcher-chosen personal literature library to the reliable daily arXiv workflow so confirmed library-derived directions surface valuable papers that manual topics miss, with inspectable discovery sources and abstract-level personal-novelty evidence.

## Success criteria

- [ ] An Obsidian desktop user can select one personal literature library inside or outside the Vault, review the exact processing scope and configured model endpoint, authorize metadata-and-abstract processing, and later revoke that authorization.
- [ ] The product incrementally builds and reloads a paper-level library catalog from identifiable PDFs and supported paper metadata without requiring full-text parsing or prior library organization; unresolved or unrelated files do not block usable entries.
- [ ] The product proposes research directions and small representative sets from the catalog, and only researcher-confirmed, corrected, merged, or disabled directions affect discovery.
- [ ] Daily discovery uses the union of manual topics and confirmed library directions without duplicating papers or regressing the existing manual-only workflow.
- [ ] Every personalized daily-report entry shows its discovery source; library-derived entries name the relevant direction and representative prior papers.
- [ ] Library-derived entries provide validated, abstract-level personal novelty as a difference type, comparison basis, evidence depth, and bounded explanation without implying full-text findings.
- [ ] Dogfood comparison demonstrates at least one researcher-accepted, library-derived paper that the same manual topics would have missed, with an understandable reason.
- [ ] Host-neutral catalog, profile, discovery, and evidence semantics live in Core; focused adversarial tests and the complete repository quality suite pass.

## Non-goals

- Autonomous Agent planning or reason-act-observe loops.
- Full-library chunking, vector-database RAG, or automatic full-text comparison.
- General-purpose chat, MCP, or a developer-facing Agent platform.
- Reading candidates, periodic direction reviews, or post-reading dispositions.
- CLI product parity in this initiative.
- Automatic filesystem watchers, background profile refresh, or Zotero/JabRef-specific integrations.

## Constraints

- Follow `CONTEXT.md` and ADR 0004; manual topics remain an explicit discovery source and combine with confirmed library directions by union.
- The first product host is the desktop-only Obsidian plugin, while reusable domain semantics remain host-neutral in `@arxiv-daily/core`.
- External-library access is scoped, read-only, and incapable of writing, deleting, or escaping the researcher-selected root.
- Local cataloging and model processing are separate actions. Consent binds library scope, eligible content types, processing depth, and effective endpoint identity; endpoint or depth changes require renewed confirmation.
- By default, only identifiable paper files are eligible; drafts, notes, and unrelated files are ignored unless a future decision explicitly includes them.
- Metadata- and abstract-level evidence must remain visibly distinct from full-text evidence.
- Existing manual-only users, daily commit semantics, checkpoints, cancellation, and deterministic Markdown rendering remain compatible.
- Do not commit or push without explicit user instruction.

## Phases

1. P1 — the desktop plugin proves safe, consent-bound read-only access to one Vault-internal or external library root without changing daily behavior — status: done
2. P2 — a durable paper-level catalog incrementally identifies and reloads usable library papers while isolating unresolved and unrelated files — status: done
3. P3 — researcher-reviewed directions and representative sets become the only library-derived inputs eligible for daily discovery — status: done
4. P4 — daily filtering combines manual topics and confirmed directions with complete, visible discovery provenance — status: done
5. P5 — personalized entries add validated abstract-level novelty evidence to the deterministic daily experience — status: pending
6. P6 — operational hardening and dogfood comparison demonstrate valuable discoveries missed by manual topics and pass full verification — status: pending

## Open questions

- Should email digest projection include discovery source and personal novelty in this initiative, or only the Markdown daily report and Dashboard?
