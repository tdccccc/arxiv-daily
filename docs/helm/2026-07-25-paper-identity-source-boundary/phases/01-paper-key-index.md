# P1 — paper-key-index

<!-- Filename 01-paper-key-index.md ↔ P1 -->
goal_ref: ../goal.md
status: done

## Outcome

`papers.json` uses stable `paperKey` values of the form `source:externalId`, legacy bare arXiv ID keys load and upgrade on write (schema 3 → 4), and on-disk note/PDF paths remain short arXiv stems.

## Assumptions

- Current schema is `PAPER_INBOX_SCHEMA_VERSION = 3` with `papers` keyed by bare modern arXiv ID and `source: "arxiv"` on each entry (`packages/core/src/services/paper-index.ts`).
- Call sites can be updated behind a small key helper without rewriting Dashboard UX.
- Users accept silent upgrade on next index save; no mandatory CLI migrate step in P1.
- arXiv version suffixes (e.g. `v2`) stay normalized the same way `modernArxivResources` already does for IDs.

## Locked decisions (grill)

| Topic | Decision |
|---|---|
| Key literal | `source:externalId`; **source lowercase** `[a-z0-9_]+`; arXiv externalId via existing modern-id normalization |
| Schema | **Bump to 4**; load schema ≤3 upgrades bare keys; save writes 4 only |
| Entry fields | **`paperKey` + `source` + `externalId`**, keep **`arxivId`** as alias (`arxivId === externalId` for arXiv) |
| P1 scope | **core index + compatible store APIs + tests**; Dashboard only if compile breaks; no UI paperKey chrome |

## Approach

1. Define `PaperKey` helpers: `formatPaperKey`, `parsePaperKey`, `paperKeyFromArxivId` / bare-id accept, validation errors.
2. `PAPER_INBOX_SCHEMA_VERSION = 4`. On load: rewrite bare arXiv map keys → `arxiv:<id>`; fill `paperKey`, `source`, `externalId`, keep `arxivId`.
3. Path helpers use **externalId** (short stem), never the full paperKey string.
4. Store get/upsert/star/detail: accept `paperKey` **or** bare arXiv id (compat); normalize before map access.
5. Minimal call-site fixes outside core only when typecheck requires it.
6. No SourceAdapter / Fake / email in this phase.

## Tasks

- [x] Implement `PaperKey` helpers (parse, format, arXiv bare-id upgrade, invalid key errors).
- [x] Schema 4 load-time normalization: bare arXiv keys → `arxiv:<id>`; populate paperKey/source/externalId/arxivId.
- [x] Save path writes only schema 4 + normalized keys; atomic write unchanged.
- [x] Path derivation stays on short externalId; comment the contract in code.
- [x] Update `PaperIndexStore` APIs for paperKey with bare-arXiv compatibility.
- [x] Unit tests: legacy schema3 load, rewrite-on-save, path stems, invalid keys, arxivId alias.
- [x] Core typecheck + paper-index tests green; fix only compile-breaking external call sites.

## Verification

- Load a fixture `papers.json` with bare arXiv keys → in-memory keys are `arxiv:…`.
- Save → on-disk keys are normalized; `paperPath` / `pdfPath` still short-id relative paths.
- Existing arXiv ID commands (summarize by id, open note) still work with bare ids via compatibility.
- `packages/core` paper-index tests and typecheck pass.

## Abort / reshape triggers

- If Dashboard, history sync, or run pipeline require a full SourceAdapter before any key migration compiles cleanly → L2: shrink P1 to helpers + load normalize only, defer store API churn.
- If multi-device sync of half-upgraded indexes proves unsafe under read-compat → L2: add explicit migrate + backup step.
- If product intent shifts away from multi-source paper keys → L3: stop and revise goal.
