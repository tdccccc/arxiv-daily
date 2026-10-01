# ADR 0011: Reading candidates and reading dispositions are dropped, not deferred

Status: Accepted (2026-08-31, recording the 2026-08-31 removal)

Related: ADR 0004 (personal-library-guided discovery), whose steps 7 and 9 and "later initiatives" list this decision closes.

## Context

ADR 0004 sketched a discovery loop beyond the daily report: save a discovered paper as a **reading candidate**, revisit saved candidates grouped by confirmed direction, and feed lightweight **reading dispositions** back into later discovery. It scoped all three out of the first initiative and listed them as later initiatives.

They were built anyway — domain and store in core, a review modal, a Dashboard save action, a review command — and reached the main branch after the 0.4.6 release. Real acceptance then showed the feature could not be triggered at all in the test library: the save action renders only on papers carrying a discovery source, and no report had one, because no manual topic and no confirmed direction existed to produce it.

The researcher judged the loop not worth its cost and stopped it. Because none of it had shipped in any release tag (0.4.3 through 0.4.6 all predate it), removing it cost nothing externally, whereas keeping it meant publishing its storage format in the next release and carrying that format as a compatibility obligation.

## Decision

### 1. Reading candidates, direction review of candidates, and reading dispositions are dropped

Not deferred, not backlogged. ADR 0004's steps 7 and 9 are closed as will-not-do. Their code was removed before any release contained it.

### 2. The research signal is confirmation and correction of directions

Refinement of library-guided discovery comes from the researcher confirming, correcting, disabling, and merging proposed directions — and from the incremental suggestions of ADR 0007. Reading feedback is not a source of research signal.

### 3. Discovery source stays

The per-paper record of why a paper entered a report is retained. It serves the explainability of the daily report itself and was never specific to reading candidates.

## Consequences

- The glossary terms *reading candidate*, *direction review*, and *reading disposition* are removed; *direction suggestion* remains and now carries the whole of the review queue.
- ADR 0004 keeps its original text as the record of what was planned; this ADR is where its steps 7 and 9 are resolved.
- Reversing this means designing the storage format afresh. That is the intended cost: the format was withdrawn precisely so it would not become a compatibility obligation. The removed implementation remains in history if it is ever wanted as a starting point.
- A test library that once saved candidates may hold orphaned documents under its index directory. Nothing reads or writes them; no released version ever created them.
