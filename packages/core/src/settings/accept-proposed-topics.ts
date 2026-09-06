import type { PersonalLibraryProposedTopic } from "../library/personal-library-interest-profile";
import { normalizeTopic } from "./topics";
import type { Topic } from "./types";

/**
 * Accepting a library proposal writes topics into product settings rather than
 * into a document of its own (ADR 0014 §1 and its last consequence): whatever
 * guards settings writes has to cover this path too.
 *
 * `normalizeTopic` stays the single entry point for producing a topic — this
 * module sits on top of it and adds the one thing it does not do, because it
 * never had to before: turn a display name into a machine tag that is unique
 * across the settings it is about to join. Duplicate tags are a validation
 * error (`validateFilterConfig`), so a proposal accepted without this would
 * write settings that refuse to run.
 */

/** Longest derived tag before the uniqueness suffix is appended. */
export const MAX_DERIVED_TOPIC_TAG_LENGTH = 40 as const;

/**
 * Tag characters are the ones already legal in a report heading and in a
 * filter prompt's `tag: directions` line. Anything else in a display name —
 * spaces, punctuation, CJK, emoji — collapses to a separator rather than
 * being transliterated, because a wrong transliteration is worse than a
 * shorter tag the researcher can rename.
 */
function slugify(name: string): string {
  const slug = name
    .normalize("NFKD")
    .replace(/[̀-ͯ]/gu, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/gu, "-")
    .replace(/^-+|-+$/gu, "")
    .slice(0, MAX_DERIVED_TOPIC_TAG_LENGTH)
    .replace(/-+$/gu, "");
  return slug;
}

/**
 * A name that slugifies to nothing — every character outside the tag alphabet,
 * which a Chinese or Japanese topic name does entirely — still needs a tag.
 * Falling back to an ordinal keeps the topic usable and visibly machine-made,
 * so renaming it reads as the obvious next step.
 */
function fallbackTag(ordinal: number): string {
  return `topic-${ordinal + 1}`;
}

function uniqueTag(base: string, taken: Set<string>): string {
  if (!taken.has(base)) return base;
  // Deterministic, and stable under re-acceptance of the same proposal: the
  // first free ordinal, not a random or time-based suffix.
  for (let suffix = 2; ; suffix += 1) {
    const candidate = `${base}-${suffix}`;
    if (!taken.has(candidate)) return candidate;
  }
}

/**
 * How a researcher tells two topics apart in settings: by the name on the
 * report heading, whatever its casing or surrounding space. The proposal
 * survives acceptance so the rest can be accepted later, which means the same
 * topic can be offered again — matching on this key is what keeps a second
 * accept from appending a second copy of it.
 */
export function topicNameKey(name: string): string {
  return name.trim().toLocaleLowerCase();
}

export interface AcceptProposedTopicsInput {
  /** Proposed topics the researcher kept, in the order they were reviewed. */
  topics: readonly Pick<PersonalLibraryProposedTopic, "suggestedName" | "directions">[];
  /** Topics already in settings; their names and tags are both reserved. */
  existingTopics: readonly Pick<Topic, "tag" | "name">[];
  /** Whether accepted topics request full paper notes. */
  detail?: boolean;
}

/**
 * Turns kept proposed topics into settings topics, skipping any whose name is
 * already in settings. Every result goes through `normalizeTopic`, so the
 * `description` rollback shadow (ADR 0012 §1) is maintained by its single
 * writer and never by this module.
 *
 * Directions carry `origin: "library"` — the first producer of that origin,
 * which P1 added to the schema with nothing yet writing it.
 */
export function acceptProposedTopics(input: AcceptProposedTopicsInput): Topic[] {
  const taken = new Set(input.existingTopics.map(({ tag }) => tag.trim()).filter(Boolean));
  const names = new Set(
    input.existingTopics.map(({ name }) => topicNameKey(name)).filter(Boolean),
  );
  const accepted: Topic[] = [];
  for (let ordinal = 0; ordinal < input.topics.length; ordinal += 1) {
    const proposed = input.topics[ordinal]!;
    const name = proposed.suggestedName.trim();
    // Accepting the same proposal twice — or a proposal that names a topic the
    // researcher already keeps — has to be a no-op, not a duplicate section in
    // every daily report from then on.
    if (names.has(topicNameKey(name))) continue;
    names.add(topicNameKey(name));
    const base = slugify(name) || fallbackTag(ordinal);
    const tag = uniqueTag(base, taken);
    taken.add(tag);
    accepted.push(normalizeTopic({
      name,
      tag,
      detail: input.detail === true,
      directions: proposed.directions.map(({ text }) => ({ text, origin: "library" })),
    }));
  }
  return accepted;
}
