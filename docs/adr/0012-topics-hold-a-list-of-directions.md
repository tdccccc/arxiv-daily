# ADR 0012: A research topic holds a list of one-line directions

Status: Accepted (2026-09-01 design session)

Related: ADR 0004 (personal-library-guided discovery); ADR 0009 (direction synthesis); ADR 0010 (confirmed directions satisfy the daily run), whose separation of topics and directions this supersedes.

## Context

A research topic today is one flat record: a display `name`, a machine `tag`, one `description`, and a `detail` flag. Only the description takes part in classification — the filter prompt is built from `tag: description`, so the name is a label and nothing more.

Library-derived research directions live in a separate document, the confirmed interest profile, and drive a second classifier of their own. The two are unioned at filter time.

Two problems converge:

- **The one-topic-per-direction model does not scale.** As a library grows its clustering yields more directions, and giving each its own topic fills the settings page long before the library is large. A researcher does not have forty topics; they have a field and forty threads running inside it.
- **Two homes for one idea.** A researcher describing their interests writes topics; the library proposes directions; the two lists look alike, live in different places, and are edited by different surfaces. Nothing in the product explains why.

Classification cost follows the same shape: the personalized classifier chunks directions twelve at a time, so a long flat list of directions multiplies calls, while directions grouped under a few topics do not.

## Decision

### 1. A topic's description becomes a list of directions

`name`, `tag`, and `detail` keep their present jobs — display heading, machine identifier and report grouping key, and whether matched papers get a full paper note. The single `description` string is replaced by an ordered list of **directions**: the specific threads running inside that topic.

### 2. A direction is one line of text

One editable line, with an identity and its origin recorded alongside. Not a name plus a description, and not a nested cue list: the point of the change is that dozens of directions stay legible in the settings page, which two-line entries would defeat. A direction proposed from the library folds its discovery cues into that one line when it is accepted.

### 3. The container keeps the name "topic"

`topics` is a released settings key that also appears in the CLI configuration, the README, and user documentation. Renaming it buys clarity that a sharpened definition buys just as well, and costs a migration across all of them.

### 4. An accepted library direction keeps no evidence

Accepting a proposed direction takes its text and nothing else. It becomes an ordinary direction, indistinguishable in behaviour from one typed by hand.

## Consequences

- The settings page holds one list of interests, and the library's role becomes proposing lines for it rather than maintaining a parallel record.
- **The confirmed interest profile document retires.** Its whole job was to carry evidence, eligibility, lineage, and merge history for accepted directions; with no evidence retained, none of that has a subject. Direction *candidates* still need a durable home until they are accepted or discarded.
- **Eligibility and staleness detection retire with it.** Nothing will notice that a direction's founding papers left the library. A direction is text, and text does not go stale on its own — the researcher edits or deletes it.
- **Personal novelty loses its stated comparison basis.** ADR 0004 defines novelty relative to a named representative set, which no longer exists. It does not follow that novelty must go: the comparison basis can become the most similar papers in the library, fetched from the retrieval index on demand. Which of those to build is an open decision, not settled here.
- Existing topics migrate by their `description` becoming the first direction. The mapping is reversible while a topic holds exactly one hand-written direction, which every migrated topic initially does.
- The filter prompt contract changes, so cached daily filter results are invalidated once. How the topic name participates in classification — a hard gate that drops out-of-field papers before matching, or a label as it is today — is deliberately left open.
