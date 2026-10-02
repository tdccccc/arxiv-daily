# ADR 0010: Confirmed library directions satisfy the daily run on their own

Status: Accepted (2026-08-31 design session)

Related: ADR 0004 (personal-library-guided discovery); ADR 0009 (direction synthesis across clusters).

## Context

Running a daily report requires the configuration check to pass, and that check requires at least one manually written research topic. It reads product settings only; confirmed library directions are invisible to it.

The pipeline itself has no such requirement. Paper filtering takes the **union** of two independent classifiers: the manual-topic classifier and the personalized direction classifier. With zero topics the manual side contributes nothing and the personalized side still emits its matches under the `personal-library` category. The engine can already produce a library-only report; only the pre-run check refuses to start one.

The consequence is that the personal literature library is structurally a supplement to hand-written topics rather than a discovery source in its own right — a researcher must first describe their interests by hand before the library they already own is allowed to contribute anything. That contradicts the product intent of deriving research directions from the accumulated library.

## Decision

### 1. Either source satisfies the check

The pre-run configuration check is satisfied by at least one manual topic **or** at least one eligible confirmed library direction. With neither, it still refuses and says so.

### 2. The check takes library state as an explicit input

Confirmed-direction state is passed into the check rather than read from product settings. The CLI product has no personal library and passes none, so its behavior is unchanged: the CLI continues to require a manual topic.

### 3. Report structure is unchanged

A library-only report keeps the existing shape: one `Library-guided discoveries` section listing the papers, each carrying its own discovery source. Directions do not become report sections. Grouping by direction remains open for later evidence from real reports.

## Consequences

- A researcher with an indexed library and confirmed directions can run daily reports without writing a single topic by hand.
- Reverting this is a breaking change for anyone running with zero topics, which is why it is recorded here rather than treated as a validation tweak.
- Discovery source stays the explanation for every paper: with no manual topics, every paper in the report names the confirmed direction that selected it.
- A researcher whose directions all become ineligible (library folder changed, representative papers removed) and who has no manual topics returns to a refusing check. The message must name that cause rather than only asking for a topic.
