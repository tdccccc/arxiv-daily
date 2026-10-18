# Journal — filter quality evaluation

## 2026-10-15 — L1 adjust (P1 Chunk 2)

- evidence: owner review of the delegate's Chunk 2 found that both `no_matches` completions (every candidate skipped; every kept paper ignored) returned before the record was written — the days recall evaluation needs most.
- change: Chunk 2 extended to record those days; failure paths still write nothing.
- disposition: kept; tests added for both paths.
- next: Chunk 3 host wiring.

## 2026-10-15 — L1 adjust (P1 Chunk 5, from the real run)

- evidence: user-consented real CLI run (model `gpt-6-luna`, astro-ph.CO + astro-ph.GA, 2026-10-14, temp vault) fetched ~100 candidates but recorded 3. The filter prompt lets the model omit irrelevant papers and the decoder accepts a subset, so omitted papers vanished from the record. The same run hit arXiv HTTP 429 during abstract enrichment and classified on titles only.
- change: added Chunk 5 — record every offered paper (omission counts as skip) and `hadAbstract` per candidate; record schema v2.
- disposition: kept; rerun recorded 100/100 candidates (~30 KB), retention stays 180 days.
- next: P1 done.

## 2026-10-15 — note (real-run observations for later phases)

- Two real runs of the same date and config selected 3 then 2 papers (a 68-score CSST weak-lensing paper dropped out): temperature 0 does not make the filter repeatable on this endpoint. P2's rerun mode should support repeated runs so variance is visible beside precision/recall.
- Both runs classified on titles only (arXiv 429 on enrichment). Evaluation must separate or exclude `hadAbstract=false` decisions; P2 export fetches abstracts itself and needs backoff/caching for 429.
- All structured summaries in both runs fell back to the raw abstract (`validation-exhausted` after 3 attempts) with `gpt-6-luna`: no Chinese structured summary was produced. Out of scope here (summary evaluation is a non-goal), but it is concrete evidence for the deferred summary-quality initiative.

## 2026-10-15 — decision (P3 inputs)

- evidence: user confirmed the trial topics (Photo-z; Galaxy Cluster) are their real interests, and consented to backfilling five announcement days with real CLI runs for the first label sheet.
- change: P3 started; the two out-of-scope findings (arXiv API throttling leading to titles-only filtering; every structured summary falling back with `gpt-6-luna`) go to read-only investigations at the user's request, outside this initiative's phases.
- disposition: none yet.
- next: backfill 2026-10-08/09/12/13, export the sheet.

## 2026-10-18 — handoff and close

- Handoff: the owner session from 2026-10-15 is gone. Its branch `feat/quality-evaluation` merged as PR #94, and the branch and worktree were deleted, so the Constraints line about them is historical. This session took over as owner.
- Evidence:
  - The researcher labelled all 48 rows.
  - Recorded decisions scored precision 37.5% and recall 100%.
  - A gpt-6.1-sol rerun with abstracts (3 runs, all identical) scored precision 60% and recall 100%.
  - Details are in P3 Chunks 2–3.
- Change: P3 done; success criterion 5 ticked; goal closed as done.
- Follow-ups worth their own initiatives:
  - (a) Two high-scoring false positives remain at score 95, so a score threshold cannot remove them; a prompt change would be needed. Rerun with this sheet before and after.
  - (b) gpt-6.1-sol sometimes omits skipped papers from its answer. Check how the daily filter treats omissions.
  - (c) There are no photo-z positives in the sheet yet, so a future sheet should include more photo-z candidates.
