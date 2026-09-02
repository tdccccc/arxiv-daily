# ADR 0014: Library proposals file themselves into topics

Status: Accepted (2026-09-02 design session)

Related: ADR 0012 (topics hold a list of directions), which creates the question this answers; ADR 0009 (direction synthesis across clusters).

## Context

Once directions live inside topics, every direction the library proposes needs a home. The previous model never asked: confirmed directions lived in a document of their own, so nothing had to decide where they belonged.

Two situations differ sharply.

**First run.** A researcher with a library and no topics at all. Asking which topic each proposal belongs to is unanswerable — there are none — and requiring them to invent topics by hand first defeats the point of deriving directions from the library they already have.

**Afterwards.** A researcher with topics that already describe their work, whose library grows and yields a handful of new candidates. Here the topics are the best available evidence of where a new direction belongs, and choosing by hand for each candidate reintroduces exactly the per-candidate friction that bulk acceptance removed.

The clusterer can already serve both: its `relativeStopRatio` stops merging when the next edge falls below a fraction of the strongest one, so lowering it yields fewer, coarser clusters. Two passes over the same vectors give a coarse level and a fine level without new machinery.

## Decision

### 1. The first scan proposes whole topics

Coarse clustering yields the topics, fine clustering within each yields that topic's directions. The researcher reviews and accepts a structure, not a pile of unattached lines. A proposed topic carries a suggested name; its machine tag is derived from that name and made unique.

### 2. Later candidates are filed by similarity, and the researcher can move them

A new candidate is proposed into the existing topic it most resembles. The suggestion is visible and changeable before acceptance; nothing is filed silently.

### 3. A candidate resembling nothing proposes a new topic

Below a similarity threshold, a candidate arrives with a suggested new topic rather than being forced into the nearest existing one. A library's subject genuinely drifts — a researcher who starts collecting a second field should see that field appear, not watch it contaminate an unrelated topic.

This matters more than tidiness once the topic name is used to exclude other fields' papers: a topic polluted by unrelated directions stops being a usable filter.

## Consequences

- A researcher can go from an indexed library to a working set of topics without writing one by hand, which is the shortest path the product has ever offered to its own premise.
- Two placement paths exist and both must be built; the first-run path is not a special case of the second, because it has no topics to compare against.
- **Two thresholds now shape what the researcher sees**: the coarse clustering ratio that decides how many topics a first scan proposes, and the similarity floor below which a candidate proposes its own topic. Set the floor too low and drift is missed; too high and the settings page fills with near-duplicate topics. Neither has a defensible default yet — both need measurement against a real library, and until then they are tuning knobs, not settled behaviour.
- Accepting a proposed topic writes a new topic into product settings, so acceptance now changes settings rather than a separate document. Whatever guards settings writes has to cover it.
