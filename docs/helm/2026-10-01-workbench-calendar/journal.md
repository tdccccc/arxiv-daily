# Journal

## 2026-10-02T00:15:05+08:00 — Calendar accepted

The user approved a calendar modeled on the Obsidian Dashboard. A new focused initiative extended the already accepted reading workbench without reopening or invalidating its completed phases.

The calendar reuses existing date arithmetic and reads full persistent run state together with actual Markdown. Obsidian-specific scheduling windows remain in its host; the CLI keeps its existing generation behavior. Browsing does not infer upstream publication availability, recover state, or start jobs.

API and UI changes followed observed Red→Green. Shared helper extraction retained its Green baseline. Browser review accepted the compact sidebar, right-side date state, explicit generation form and mobile collapse. A discovered same-month refresh/selection race was reproduced and fixed before acceptance. All goal criteria are met; no waiver.

Evidence: 183 CLI tests, seven package tests, original 40-test plugin calendar baseline (32 repeated in final integration), workspace build/typechecks, boundaries, product inventory, submission check and strict plugin validation pass; lint has 21 pre-existing warnings. Screenshot and read-only fixture evidence is recorded in the phase. All changes remain local to the existing worktree, committed and unpublished.

## 2026-10-02T00:20:27+08:00 — Follow-up P2 readability

The user accepted the calendar functionality but found it too small and requested a wider sidebar, colored cells and visible paper counts. Keep P1 accepted; P2 refines presentation and adds optional count recovery from the existing Paper Index. Navigation and generation semantics remain unchanged.
