# ADR 0014: Library proposals file themselves into topics

Status: Accepted (2026-09-02; first-scan organization revised 2026-09-05)

Related: ADR 0012 (topics hold a list of directions), which creates the question this answers; ADR 0009 (direction synthesis across clusters).

## Context

Once directions live inside topics, every direction the library proposes needs a home. The previous model never asked: confirmed directions lived in a document of their own, so nothing had to decide where they belonged.

Two situations differ sharply.

**First run.** A researcher with a library and no topics at all. Asking which topic each proposal belongs to is unanswerable — there are none — and requiring them to invent topics by hand first defeats the point of deriving directions from the library they already have.

**Afterwards.** A researcher with topics that already describe their work, whose library grows and yields a handful of new candidates. Here the topics are the best available evidence of where a new direction belongs, and choosing by hand for each candidate reintroduces exactly the per-candidate friction that bulk acceptance removed.

The first implementation used coarse clusters as topics and fine clusters as directions. Real-library measurement invalidated that path: single linkage chained almost the whole corpus together, while average linkage separated it into ten topics and 46 overly narrow directions. Changing the fine quantile barely reduced direction count. Similarity successfully identified related papers, but cluster boundaries were not a useful boundary for a researcher's standing interests.

## Decision

### 1. The first scan proposes whole topics

One tight clustering pass supplies evidence groups. A model sees those groups together and organizes them into 2–4 topics, each with 1–2 broad, one-line directions; one evidence group may yield one topic. It also supplies suggested names. Quantity and level-of-abstraction constraints guide it without examples copied from the researcher's settings. This replaces per-cluster extraction, synthesis (ADR 0009), and per-topic naming calls.

Every evidence group must be assigned exactly once. Direction membership is the full union of assigned groups, while representative papers must come from those groups. Invalid or incomplete assignments are retried within the generation budget and then rejected, never silently repaired by dropping evidence. Ungrouped papers remain visible as uncovered evidence.

The researcher reviews and accepts a structure. Topics are sorted by distinct paper coverage, the two largest are selected initially, and other topics are marked optional. Collapsed rows show paper and direction counts; selection and editing remain available. A topic's machine tag is derived from its suggested or edited name and made unique.

### 2. Later candidates are filed by similarity, and the researcher can move them

A new candidate is proposed into the existing topic it most resembles. The suggestion is visible and changeable before acceptance; nothing is filed silently.

### 3. A candidate resembling nothing proposes a new topic

Below a similarity threshold, a candidate arrives with a suggested new topic rather than being forced into the nearest existing one. A library's subject genuinely drifts — a researcher who starts collecting a second field should see that field appear, not watch it contaminate an unrelated topic.

This matters more than tidiness once the topic name is used to exclude other fields' papers: a topic polluted by unrelated directions stops being a usable filter.

## Consequences

- A researcher can go from an indexed library to a working set of topics without writing one by hand, which is the shortest path the product has ever offered to its own premise.
- Two placement paths exist and both must be built; the first-run path is not a special case of the second, because it has no topics to compare against.
- The first scan no longer uses a coarse similarity threshold to determine topic count. Tight clustering remains an evidence-grouping parameter, recorded in the generation contract; the later-placement similarity floor still needs measurement against a real library. Set that floor too low and drift is missed; too high and settings fill with near-duplicate topics.
- Accepting a proposed topic writes a new topic into product settings, so acceptance now changes settings rather than a separate document. Whatever guards settings writes has to cover it.
- The review page shows uncovered evidence as a single count rather than the per-paper list this decision originally described; the researcher found the list noisy. The count still says how much of the library a proposal does not speak for, but those papers are named nowhere in the product now — reading which ones they are costs a regeneration or a look at the library itself.
