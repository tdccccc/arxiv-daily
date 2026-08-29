# P6 — operational hardening and dogfood comparison

goal_ref: ../goal.md
updated: 2026-08-04

## Outcome

The personalized discovery path survives real-library scale and an adversarial hardening review; a dogfood comparison against the researcher's real library demonstrates at least one researcher-accepted, library-derived discovery that the same manual topics would have missed, with an understandable reason; the email-digest open question is resolved; the complete repository quality suite and independent correctness/security reviews pass; the initiative closes with every success criterion checked.

## Assumptions

- The dogfood library is `~/Nextcloud/work/Article` (user decision 2026-08-04): 526 files, 25 arXiv-ID-identifiable, 352 author-year style, plus assorted other names — a realistic messy library. Author-year and unidentifiable files remain unresolved/unrelated and are acceptable as such.
- The email-digest open question resolves to: keep digest/email projection unchanged (user leaning, 2026-08-04). This is a documented decision, not a deferral — the data (occurrence provenance, personal novelty) is already persisted in the Paper Index and daily-report markers, so a later email projection is an isolated digest-model + render change with low rework cost.
- The dogfood run happens in the user's real Obsidian vault (`~/Nextcloud/work/Notes`) with the real library connection, consent, direction review, and daily run, operated by the user; I prepare the run kit and analyze the produced artifacts (catalog, profile, `papers.json`, daily reports). I never touch API keys or authorization state.
- Headless verification of catalog/reconciliation/eligibility against the real library is feasible through `@arxiv-daily/node-runtime` scoped source + exported Core library modules, without the Obsidian host.
- Manual topics stay as configured by the user; the comparison is manual-topics-only vs. union of manual topics + confirmed library directions on the same daily window.

## Approach

Resolve the email decision first (journaled, no code change). Then drive Core's scoped-library, catalog, and reconciliation path against the real Article directory from a headless node script to validate scale behavior (identifiable set, unresolved isolation, incremental reload reuse, no PDF-byte reads, bounded runtime) and fix whatever real data exposes. Run an adversarial hardening review over the whole personalization surface (consent/eligibility boundaries, revocation and transition aborts, catalog/profile/corruption recovery, marker integrity, privacy payload minimization) with regressions for every accepted fix. **L2 reshape 2026-08-04: widen identification** — real-library dogfood showed filename-only arXiv-ID recognition is a hard blocker (test_library: 22/86 = 25.6% identified; real Article library same ratio). User decision: parse PDF text content instead of patching filename heuristics. New identification strategy v2: (1) filename arXiv ID (kept), (2) PDF evidence extraction — decompress content streams with pako (already a core dependency), extract text literals, scan for arXiv header IDs, /Title (literal/hex/UTF-16), and XMP dc:identifier; (3) arXiv title-search fallback over HTTPS for titles without embedded IDs, accepted only under strict normalized-title matching; unidentifiable files (scanned PDFs without text layer) stay unresolved without blocking. Measured on the real test library: +23/64 files identified from embedded arXiv headers (36%), recognition rises to ~52%; scanned no-text-layer PDFs (53%) are a physical limit, custom-encoded fonts (11%) need a full PDF parser (out of scope, bundle cost not justified). Prepare the user's Obsidian dogfood run kit and analysis scripts, then jointly execute the comparison: user operates the plugin, I extract and compare discovery sources, identify candidate library-derived papers missed by manual topics, and the user accepts at least one with an understandable reason. Finish with the complete repository verification and independent reviews, technical-report handoffs per accepted chunk, journal, and staged commits without pushing.

## Tasks

- [x] Resolve the email-digest open question (keep digest/email unchanged), journal the decision with the persisted-data rationale; update goal.md open questions and mark P6 active.
- [x] Headless real-library validation: a node script opens `~/Nextcloud/work/Article` through the node-runtime scoped source, builds the catalog via Core reconciliation into a temp vault, reloads to prove fingerprint reuse, and reports identifiable / unresolved / unrelated / failed / truncated counts and runtime; fix any exposed issue with a regression test.
- [x] Adversarial hardening review of the personalization surface: consent/eligibility boundaries, revocation and transition aborts, catalog/profile/corruption and backup recovery, marker parsing integrity, and privacy payload minimization (roots, paths, PDF bytes, credentials); accept only fixes with focused regression tests.
- [x] Prepare the user's dogfood run kit: exact Obsidian steps (connect Article library → review inventory → authorize → review/confirm directions → run daily) plus analysis scripts that extract catalog stats, confirmed directions, occurrence provenance/novelty, and a manual-topics-only baseline comparison from the produced artifacts.
- [ ] Widen identification (L2 reshape 2026-08-04): implement identification strategy v2 — PDF evidence extraction (stream decompression, literal extraction, arXiv header /Title with literal/hex/UTF-16 decoding, XMP dc:identifier), arXiv title-search fallback with strict normalized-title acceptance, fingerprint v2 catalog migration, and instrumented privacy boundaries (read only for identification; title sent to arXiv API only when searching); fix the plugin scan path to read PDF evidence through the scoped source.
- [ ] Joint dogfood comparison: user operates the plugin in Obsidian; I analyze the artifacts, identify library-derived papers that the same manual topics missed, and the user accepts at least one with an understandable reason; document the comparison in the initiative docs.
- [ ] Full verification: complete Core and plugin suites with the established Core heap allowance; node-runtime and CLI regressions; workspace typecheck/build; plugin production build; ESLint; boundary check; `git diff --check`; independent correctness/security reviews of the accepted changes.
- [ ] Close P6: every accepted chunk completes its technical-report handoff (`updated`/`no-impact`), journal the phase, check all goal success criteria (waivers named), mark P6 done and goal done, and create staged local commits without pushing.

## Verification

- Email decision: goal.md open questions section updated, journal entry states the decision and why later email projection stays cheap.
- Real-library validation: the Article catalog covers all 25 arXiv-identifiable files as papers; author-year and other files land in unresolved/unrelated/failed isolation without blocking usable entries; reload reuses unchanged entries (no re-identification work); instrumented output proves no PDF bytes or absolute paths are read/persisted; runtime stays bounded (no minutes-scale hang).
- Hardening review: findings list with disposition; each accepted fix carries a regression test proving the boundary holds (e.g., revocation mid-run → zero further model calls; corrupt backup → primary recovery).
- Dogfood run kit: the user can follow the steps without further explanation; analysis scripts produce catalog stats, direction list, and per-paper discovery-source + novelty extraction from `papers.json` and daily Markdown.
- Dogfood comparison: at least one researcher-accepted library-derived paper (or explicit reasoned waiver with a named criterion) that the same manual topics would have missed, with an understandable reason naming the direction and representative basis; documented in the initiative docs.
- Full suite: all commands in the Constraints/quality list pass with observed output; independent correctness/security reviews note no open findings in the accepted scope.
- Close: goal.md status `done`, every success criterion checked or waived in journal.md, technical-report handoffs all `updated`/`no-impact`, staged commits on `docs/agent-product-strategy` without push.

## Abort / reshape triggers

- ~~If the real Article library yields too few identifiable papers for usable direction proposals, L2 reshape the dogfood scope (e.g., widen identification or narrow the demonstration claim) instead of forcing a weak comparison.~~ **Triggered 2026-08-04 → reshaped: identification widened to PDF-evidence extraction + arXiv title search (see Approach).** If the widened identification still yields too few papers for usable directions, narrow the demonstration claim instead of forcing a weak comparison.
- If the hardening review finds a consent/eligibility boundary hole, fix it before any further dogfood or verification work — do not close with a known boundary defect.
- If the user's Obsidian run is impossible (endpoint unavailable, consent blocker, vault issue), document it and L2 reshape the demonstration to the best available evidence instead of fabricating results.
- If the email decision reverses, move email projection into P6 scope and re-plan tasks instead of closing with a different resolution than the user expects.
- If real-library scale exposes performance or correctness problems that need product-level rework (not local fixes), stop and classify L1/L2 before continuing.
