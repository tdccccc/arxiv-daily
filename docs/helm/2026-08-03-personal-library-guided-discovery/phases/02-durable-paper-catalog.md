# P2 — durable paper-level catalog

goal_ref: ../goal.md
updated: 2026-08-03

## Outcome

After an explicit Obsidian desktop scan, the selected personal literature library has a durable, reloadable paper-level catalog whose usable modern-arXiv papers contain canonical metadata and abstracts, while unresolved, unrelated, missing, and temporarily failed files remain isolated and do not block the catalog; no profile, LLM, or daily behavior changes.

## Assumptions

- Modern arXiv IDs found deterministically in PDF filenames are a sufficient first identification signal to prove the catalog lifecycle; PDF Info/XMP, first-page text, DOI, and legacy arXiv IDs can be added only if real-library evidence shows filename identification is insufficient.
- arXiv Atom metadata is an acceptable paper-level enrichment source because only canonical arXiv IDs, not PDF contents, are sent to arXiv.
- A replaceable catalog projection under the configured Vault index root may persist logical root-relative filenames; the absolute library root and processing authorization remain plugin-local connection state.
- A complete, non-truncated inventory may remove missing file contributions; a truncated or failed scan must preserve prior records rather than infer deletion.
- P2 does not require reading PDF bytes, full-text parsing, LLM calls, directions, representative sets, or changes to daily discovery.

## Approach

Add a Core-owned catalog distinct from `PaperIndexStore`. A scan inventories eligible PDFs, derives a versioned observation from logical path plus safe size/mtime metadata, reuses unchanged file records, identifies modern arXiv IDs from filenames, batch-enriches canonical IDs through the existing arXiv metadata path, and atomically commits one validated catalog. Per-file outcomes (`ready`, `unresolved`, `unrelated`, `failed`) isolate bad inputs. Plugin code owns the explicit scan lifecycle, progress, cancellation, source reopening, and summary UI; the scheduler and `ArxivPipeline` remain untouched.

## Tasks

- [x] Define and test the host-neutral catalog schema, strict decoder, scan summary, paper/file outcome semantics, scope/identification fingerprint, and storage path under the existing index root.
- [x] Implement a durable catalog store with valid-backup recovery, fail-closed mutation on unreadable current/backup data, path-scoped serialization, atomic whole-document writes, and deterministic semantic revisioning.
- [ ] Extend scoped inventory with safely obtained size/mtime observations, including bounded fallback for unknown `Dirent` types, without weakening canonical-root, identity, symlink, entry, depth, cancellation, or redacted-error guarantees.
- [ ] Implement deterministic modern-arXiv PDF filename identification and incremental reconciliation: unchanged reuse, unresolved/unrelated/failed isolation, duplicate file-to-paper membership, complete-scan deletion, and truncated-scan preservation.
- [ ] Add batched arXiv metadata/abstract enrichment through an injected Core resolver, preserving prior usable records and per-file failures when IDs are absent or network enrichment is partial.
- [ ] Wire one explicit plugin scan/reload action with root-change/unload/supersession cancellation, summarized ready/unresolved/failed counts, and no requirement for model-processing authorization.
- [ ] Add adversarial, persistence, incremental, lifecycle, and no-impact regression coverage; run affected full suites, security review, technical-report handoff, and a staged commit.

## Verification

- Core tests reload the same validated catalog, recover from a valid backup, refuse unsafe future/corrupt schemas, serialize concurrent mutations, and leave the previous catalog intact after cancellation or failed persistence.
- Reconciliation tests prove unchanged files avoid re-identification/enrichment, duplicate PDFs map to one paper, unresolved/failed files do not block ready papers, missing files are removed only after a complete inventory, and truncated inventories preserve unseen prior records.
- Node-runtime tests prove size/mtime and unknown-entry fallback remain root-contained, no-symlink, bounded, cancellable, and redacted.
- Plugin tests prove scan/reload survives restart, root changes and unload cancel active scans, model authorization is not required, and malformed catalog errors are visible without leaking absolute paths.
- Instrumented tests prove no PDF bytes or unrelated files are sent to arXiv, no LLM client is invoked, and P2 does not mutate `PaperIndexStore`, daily reports, paper notes, scheduler state, email state, directions, or representative sets.
- Core, node-runtime, plugin, and CLI full tests; workspace typecheck/build; lint; boundary check; and `git diff --check` pass.

## Abort / reshape triggers

- If a representative real library yields too few modern arXiv IDs from filenames to produce a useful catalog, stop and L2 reshape the identification task around a bounded PDF Info/XMP parser instead of silently expanding to full text.
- If safe size/mtime observations require weakening the scoped filesystem boundary, use a content-independent scan version or reprocess files rather than widening access.
- If arXiv metadata enrichment cannot preserve prior usable records under partial failure, split local identification from network enrichment before wiring the plugin scan.
- If the catalog starts carrying user judgments, inferred directions, embeddings, full text, or daily discovery state, move that work to P3 or later instead of widening P2.
