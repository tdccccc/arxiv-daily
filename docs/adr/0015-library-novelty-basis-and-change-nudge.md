# ADR 0015: Novelty compares against retrieved library papers; library changes nudge a new-papers-only rescan

Status: Accepted (2026-10-15); §1–2 withdrawn 2026-10-15, §3–5 in force

Related: ADR 0004 (personal-library-guided discovery), whose §4 novelty basis §1–2 proposed to replace and then withdrew; ADR 0007 (incremental trigger on index completion), which this supersedes; ADR 0008 (opt-in remote embedding); ADR 0011 (reading candidates dropped); ADR 0012 (topics hold a list of directions), which left the novelty basis open; ADR 0013 (index titles and abstracts); ADR 0014 (library proposals file into topics).

## Context

Two library-guided features went dormant after the direction model changed, and both are premises of the product: a daily report that says what each paper adds relative to what the researcher already holds, and directions that follow the library as it grows.

**Personal novelty.** ADR 0004 §4 defines novelty relative to the matched direction's representative set. ADR 0012 §4 removed that set: an accepted direction is one line of text and keeps no evidence. The personal novelty generator still exists in core but nothing calls it. ADR 0012 recorded that the basis could become "the most similar papers in the library, fetched from the retrieval index on demand" and left the choice open. That retrieval already exists as library similarity over a title-and-abstract index (ADR 0013); the catalog stores each paper's title and abstract. Lexical (BM25) retrieval needs no embedding, so it needs no remote endpoint; remote embedding requires its own consent (ADR 0008).

**Incremental updates.** ADR 0007 had index completion trigger placement of new papers, reclustering of a buffer pool, and a review queue of attach, new, split and merge suggestions. That code remains but has had no production caller since the auto-trigger was withdrawn: it froze Obsidian for about a minute after indexing. ADR 0014 §2 then replaced vector-threshold placement with semantic placement, where the model reads the existing topics and direction text alongside the library evidence. Later scans are manual. What is missing is only a way for the researcher to learn that the library has changed enough to be worth a scan, and a scan that does not re-send the whole library.

Two constraints from earlier decisions bound the design. ADR 0014 refuses unmeasured similarity floors; and ADR 0007's own lesson is that an automatic model call after indexing is not acceptable.

## Decision

### 1. Novelty compares each digest paper with its most similar library papers

> Withdrawn (2026-10-15). See "Withdrawal of §1–2" below. The text is kept as the record of what was decided and shipped to a test library; the product does not do this.

For every paper in a daily digest, the comparison basis is the library papers most similar to that paper, retrieved on demand from the title-and-abstract index, at most 5. This applies to all digest papers, not only those selected by a direction the library proposed: a direction is text and carries no origin that matters at classification time (ADR 0012 §4), and the researcher wants to know what each paper adds relative to what they hold regardless of which line selected it.

The model receives the digest paper and those retrieved papers and either states a difference type with its comparison basis and evidence depth (ADR 0004 §4 stays in force for that expression), or judges that no retrieved paper is comparable. Evidence depth stays titles and abstracts; a novelty line must not be presented as a full-text finding.

No similarity floor is applied before or after retrieval. Whether a retrieved paper is comparable is the model's judgment, consistent with ADR 0014. Retrieval returns a ranked top-N; it does not claim that rank implies relevance.

### 2. Novelty runs only under valid library processing consent

> Withdrawn (2026-10-15). See "Withdrawal of §1–2" below.

The step sends library titles and abstracts to the configured model endpoint, so it runs only while library processing consent is valid for that endpoint and scope (ADR 0004 §7). Retrieval mode follows the library's embedding mode; lexical retrieval is available without any embedding endpoint, and a remote embedding endpoint is used only where its own consent already exists.

A paper gets no novelty line, and the digest still completes, when any of these holds: consent is absent or revoked, the library or its index is unavailable, the model judges that no retrieved paper is comparable, or the step fails for that paper. Missing novelty is never an error of the daily run and is not retried into a different basis. The step costs at most one additional model call per digest paper and keeps the bounds and validation of the existing generator. Email is unchanged; novelty appears in the daily report and the Dashboard paper row.

The daily pipeline in core does not depend on library modules. Hosts supply the retrieval as an injected port, so that the pipeline stays host-neutral and the plugin and CLI behave alike.

### 3. A local count nudges the researcher after indexing

When indexing completes, the product counts, locally and without any model call or consent, the library papers that no scan has yet accounted for. A paper is accounted for once a scan has seen it, whether it ended up covered, assigned to a proposed direction or left ungrouped. The current proposal records the papers it accounted for together with those its predecessors accounted for, so a rescan over only the new papers does not make the rest of the library look new again. This is a set comparison against the catalog; no placement or similarity computation is involved.

When that count reaches 3, the plugin tells the researcher that the library has changed and offers an update of direction suggestions. The value 3 is carried over from the buffer-pool trigger of ADR 0007. It is a nudge count chosen to avoid prompting on every single added paper; it is not a measured similarity threshold and does not claim that three papers constitute a new research thread. It can be revised on experience.

The nudge never calls a model, never starts a scan, and never changes directions or topics.

### 4. The rescan sends only the new papers

When the researcher accepts the nudge, or starts a scan at any time, the rescan sends the new papers (those counted in §3) together with the existing topics and direction text to the model. The result follows ADR 0014 §2: each new paper is covered by an identified existing direction, or assigned to a proposed direction in an existing topic, or to a proposed new topic. It becomes a new proposal and goes through the existing review, preview and acceptance flow. Nothing is filed or applied silently.

The rescan replaces the current proposal as ADR 0014 §4 describes, carrying forward the set of papers already accounted for. Replacing a proposal that still has unreviewed candidates or drafts asks the researcher first, as replacement already does. There is no separate suggestion store, no queue of suggestions, and no repeated-run command.

### 5. What is not carried forward from ADR 0007

- Placement of new papers by vector similarity, the buffer pool and local reclustering of it.
- Attach, split and merge suggestions; direction locking.
- Automatic triggering of any model call on index completion.
- A separate suggestions document with supersede hints, and a repeating command for it. Proposals and their review already cover this.

Split and merge suggestions and direction locking are not rejected on principle; they are not wanted now. A later decision may introduce them against the text-only direction model.

## Withdrawal of §1–2 (2026-10-15)

Personal novelty is withdrawn from the product before release: no pipeline stage, no novelty line in the daily report, no marker written, no Dashboard or workbench display, no disclosure of this use in the consent text. The code and acceptance scenarios added for it were removed afterwards; the commits that added them stay in history.

Why, from using it on a real library:

- **The researcher does not choose the comparison.** The papers a digest paper is compared with are picked by text retrieval over titles and abstracts. There is no way for the researcher to confirm that a given library paper is one they care to compare against, and no way to filter for that. A line can be faithful to both abstracts and still be a comparison nobody asked for.
- **The lines reduce to "a related library paper exists".** In evaluation against a real library, the lines that were written nearly always cited a single library paper, often a catalog or simulation release that the new paper happened to use, and the difference type was almost always "new task". Whatever the line said, the information was that the library holds something related. The researcher judged that not worth a daily line. Even after the prompt was tightened so the model declined most papers (in the final calibration run, 5 lines for 20 digest papers), the lines that remained rested on that single related paper.
- **Downgrading was considered and rejected.** Replacing the difference-type explanation with a plain "you hold a related library paper: <title>" line was weighed and rejected for the same reason: it states the one thing the evaluation showed, and the researcher said it was of little use.
- The step also cost about a minute added to a daily run of 20 papers (one sequential model call per paper), which was acceptable only if the lines earned it.

What was kept from this work, because it is useful without novelty:

- The lexical query is capped at the BM25 limit on distinct terms inside core search, so a pasted abstract or other long free-text query to lexical or hybrid library search no longer fails with "more than 64 unique terms". Text within the limit is searched unchanged.
- The BM25 index verification speedup (faster hashing and hoisted bucket computation). Results are byte-identical to before.

Compatibility: reports written while the feature existed contain a personal-novelty marker line and a visible quote line. Nothing parses the marker any more; those lines are ordinary report text and are neither flagged nor removed. Layout code that locates the topic-direction marker still counts the legacy marker as a preceding line pair so older reports keep their direction data. The paper index drops any stored novelty field the next time it is saved; its schema version is unchanged. Library processing consent is unaffected: its fingerprint did not change when the feature was added.

Direction suggestions and the rescan (§3–5) did not depend on novelty and are unchanged.

## Consequences

- Struck by the withdrawal of §1–2 (kept for the record, no longer in force): every digest paper carrying a comparison against the library; one extra model call per digest paper and the consent gate for that use; reliance on the model declining to compare unrelated papers; `file:` knowledge-base documents as comparison basis; marker version 2 carrying basis titles. Researchers see the daily digest exactly as before this ADR, with or without library authorization.
- The researcher is told when the library has drifted from the current proposal without any model spend or consent, and chooses whether to scan. The cost of the model call occurs only on that explicit action, and scales with the new papers rather than the library.
- The nudge count 3 is a heuristic. Too low and the nudge becomes noise; too high and a small but real shift goes unnoticed. It is stated here so that it can be argued with.
- A nudge is only meaningful relative to the current proposal; with no proposal yet, the first scan is the existing first-run flow of ADR 0014 §1.
- The dormant incremental code (placement, buffer pool, recluster, suggestion store and command) has no remaining role and can be removed; it stays in history. ADR 0007 is marked superseded; its body is kept as the record of what was planned.
- Papers removed or changed in the library do not count toward the nudge. Detecting that a direction no longer matches the library remains out of scope, as in ADR 0012.
