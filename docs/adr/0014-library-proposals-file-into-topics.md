# ADR 0014: Library proposals file themselves into topics

Status: Accepted (2026-09-02; first-scan organization revised 2026-09-05; later placement revised 2026-09-07; bounded evidence and review revised 2026-09-08)

Related: ADR 0012 (topics hold a list of directions), which creates the question this answers; ADR 0009 (direction synthesis across clusters).

## Context

Once directions live inside topics, every direction the library proposes needs a home. The previous model never asked: confirmed directions lived in a document of their own, so nothing had to decide where they belonged.

Two situations differ sharply.

**First run.** A researcher with a library and no topics at all. Asking which topic each proposal belongs to is unanswerable — there are none — and requiring them to invent topics by hand first defeats the point of deriving directions from the library they already have.

**Afterwards.** A researcher with topics that already describe their work, whose library grows and yields a handful of new candidates. Here the topics are the best available evidence of where a new direction belongs, and choosing by hand for each candidate reintroduces exactly the per-candidate friction that bulk acceptance removed.

The first implementation used coarse clusters as topics and fine clusters as directions. Real-library measurement invalidated that path: single linkage chained almost the whole corpus together, while average linkage separated it into ten topics and 46 overly narrow directions. Changing the fine quantile barely reduced direction count. Similarity successfully identified related papers, but cluster boundaries were not a useful boundary for a researcher's standing interests.

## Decision

### 1. The first scan proposes whole topics

One tight clustering pass supplies evidence groups. A model organizes them into 2–4 topics, each with 1–2 broad, one-line directions; one evidence group may yield one topic. It also supplies suggested names. Quantity and level-of-abstraction constraints guide it without examples copied from the researcher's settings. Small inputs fit one request. Larger inputs are transported in bounded batches with complete abstracts, then their evidence-backed research lines are combined in bounded reduction passes. Batch boundaries do not define topics; complete source membership stays local throughout reduction. This replaces equal truncation of every abstract, which left about 101 characters per paper in a 200-paper probe and rejected a 500-paper title envelope.

Every evidence group must be assigned exactly once. Direction membership is the full union of assigned groups, while representative papers must come from those groups. Invalid or incomplete assignments are retried within the generation budget and then rejected, never silently repaired by dropping evidence. Ungrouped papers remain visible as uncovered evidence.

The researcher reviews and accepts a structure. Topics are sorted by distinct paper coverage, the two largest are selected initially, and other topics are marked optional. Collapsed rows show paper and direction counts; selection and editing remain available. A topic's machine tag is derived from its suggested or edited name and made unique.

### 2. Later scans distinguish existing coverage from proposed changes

The model reads the existing topics and their direction text alongside the library evidence. Each evidence group is either covered by an identified existing direction or assigned to a proposed direction. Complete coverage is a successful result with no proposed additions. The partition stays complete; covered evidence is accounted for without manufacturing new interests.

The proposal retains the covering topic and direction identities, direction text, and full paper membership. A later edit or deletion makes that coverage need review; renaming a topic does not. Older proposals without this comparison basis show unverified coverage. This is proposal evidence, not a property added to accepted directions.

A proposed direction may target an existing topic. The suggestion is visible and changeable before acceptance; nothing is filed silently. Semantic placement replaces the unmeasured vector threshold originally planned here.

### 3. A candidate resembling nothing proposes a new topic

A candidate outside the existing directions' scope arrives with a suggested new topic rather than being forced into the nearest existing one. A library's subject genuinely drifts — a researcher who starts collecting a second field should see that field appear, not watch it contaminate an unrelated topic. Topic names remain display labels; accepted direction text drives classification.

### 4. Acceptance tracks directions and stable destinations

An existing destination is referenced by topic identity, so renaming does not create a second topic. A deleted destination requires the researcher to choose again; it never silently becomes a new topic. A same-name topic is not evidence that every proposed direction was accepted.

The plugin records which candidates in the current proposal have been processed and where a partially accepted topic was placed. This small receipt and the resulting settings are saved together. Proposal storage is not part of the acceptance transaction. The receipt prevents an old proposal from restoring a direction the researcher subsequently edited or removed; accepted directions themselves retain no library evidence or special classifier behavior. Replacing the current proposal retires its unconfirmed edits and its review state.

Review drafts survive selection changes, view changes and failed saves. Accepting saves selected edits before applying the existing settings transaction; an edit failure stops acceptance. Local-file representatives may be selected from that proposal's known evidence. Papers already assigned to existing coverage are not offered as representatives for new directions.

### 5. Inspect the analysis and preview a direction

Research settings offers the library path directly. The review window also provides a dated library overview with topics, directions, coverage and expandable unassigned papers. Paper titles open the scoped local PDF. A direction preview classifies a bounded library sample using the daily filter contract and reports known category gaps without changing subscriptions. Changing the direction, refreshing the review or retrying a preview invalidates the prior result; a displayed preview also checks the current category selection.

## Consequences

- A researcher can go from an indexed library to a working set of topics without writing one by hand, which is the shortest path the product has ever offered to its own premise.
- First and subsequent scans share one organization contract; an empty existing-topic list selects the first-run behavior.
- Tight clustering remains an evidence-grouping parameter, recorded in the generation contract. Placement uses the existing direction text and researcher review, without adding an unmeasured similarity floor.
- Accepting a proposed topic writes a new topic into product settings, so acceptance now changes settings rather than a separate document. Whatever guards settings writes has to cover it.
- The proposal page keeps uncovered evidence as a single count; its expandable paper list lives in the library overview, preserving the quiet default while making omissions inspectable.
- Complete abstract batches cost more model calls than a truncated single request. Structure and source membership are validated at each stage; semantic quality must also be checked against actual libraries.
