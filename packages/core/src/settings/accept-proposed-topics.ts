import type { PersonalLibraryProposedTopic } from "../library/personal-library-interest-profile";
import { normalizeTopic } from "./topics";
import type { Topic } from "./types";

/**
 * Accepting a library proposal writes topics into product settings rather than
 * into a document of its own (ADR 0014 §1 and its last consequence): whatever
 * guards settings writes has to cover this path too.
 *
 * The result is a private candidate containing both complete topics and its
 * acceptance receipt. The host saves them in one transaction. Stable topic
 * destinations survive renames, and processed candidate identities protect
 * later edits and deletions. New topics also get unique classifier tags.
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
 * A display-name match can suggest a destination for a new proposal. It does
 * not say whether any of that proposal's directions have been accepted.
 */
export function topicNameKey(name: string): string {
  return name.trim().toLocaleLowerCase();
}

/** Saved with settings, never with the evidence document (ADR 0014 §4). */
export interface ProposalAcceptanceReceipt {
  proposalId: string;
  scopeFingerprint: string;
  topicTargets: Record<string, string>;
  processedCandidateIds: string[];
}

type AcceptableTopic = Pick<PersonalLibraryProposedTopic, "id" | "suggestedName" | "directions"> & {
  targetTopicId?: string;
};

export interface AcceptProposedTopicsInput {
  proposalId: string;
  scopeFingerprint: string;
  /** Only the directions the researcher selected, in their reviewed order. */
  topics: readonly AcceptableTopic[];
  existingTopics: readonly Topic[];
  acceptance?: ProposalAcceptanceReceipt | null;
  /** Applies only to a new topic, never changes an existing destination. */
  detail?: boolean;
}

export interface AcceptProposedTopicsResult {
  /** Complete candidate settings; the caller publishes them after persistence. */
  topics: Topic[];
  acceptance: ProposalAcceptanceReceipt;
  addedTopicCount: number;
  addedDirectionCount: number;
}

export class ProposalAcceptanceError extends Error {
  constructor(readonly code: "target-missing" | "target-ambiguous" | "invalid-input", message: string) {
    super(message);
    this.name = "ProposalAcceptanceError";
  }
}

export function directionTextKey(text: string): string {
  return text.trim().replace(/\s+/gu, " ").toLocaleLowerCase();
}

export function matchingProposalAcceptance(
  proposal: { proposalId: string; scopeFingerprint: string },
  receipt: ProposalAcceptanceReceipt | null | undefined,
): ProposalAcceptanceReceipt | null {
  return receipt?.proposalId === proposal.proposalId && receipt.scopeFingerprint === proposal.scopeFingerprint
    ? receipt : null;
}

/** The same destination rule is used by acceptance and the review preview. */
export function resolveProposedTopicTarget(
  proposed: Pick<AcceptableTopic, "id" | "suggestedName" | "targetTopicId">,
  topics: readonly Topic[],
  acceptance?: ProposalAcceptanceReceipt | null,
): Topic | null {
  const targetId = proposed.targetTopicId ?? acceptance?.topicTargets[proposed.id];
  if (targetId) {
    const target = topics.find(({ id }) => id === targetId);
    if (!target) throw new ProposalAcceptanceError("target-missing", "The destination topic was removed. Choose a destination again.");
    return target;
  }
  const sameName = topics.filter(({ name }) => topicNameKey(name) === topicNameKey(proposed.suggestedName));
  if (sameName.length > 1) {
    throw new ProposalAcceptanceError("target-ambiguous", "Several topics have this name. Choose a destination explicitly.");
  }
  return sameName[0] ?? null;
}

/**
 * A pure settings transaction: accepting a subset leaves the rest available;
 * replay never overwrites edits or resurrects an already processed direction.
 * Text equivalence supplements identity without changing manual directions.
 */
export function acceptProposedTopics(input: AcceptProposedTopicsInput): AcceptProposedTopicsResult {
  const previous = matchingProposalAcceptance(input, input.acceptance);
  const acceptance: ProposalAcceptanceReceipt = {
    proposalId: input.proposalId,
    scopeFingerprint: input.scopeFingerprint,
    topicTargets: { ...previous?.topicTargets },
    processedCandidateIds: [...(previous?.processedCandidateIds ?? [])],
  };
  const processed = new Set(acceptance.processedCandidateIds);
  const topics = input.existingTopics.map((topic) => ({
    ...topic, directions: topic.directions.map((direction) => ({ ...direction })),
  }));
  const taken = new Set(topics.map(({ tag }) => tag.trim()).filter(Boolean));
  let addedTopicCount = 0;
  let addedDirectionCount = 0;
  for (const [ordinal, proposed] of input.topics.entries()) {
    const pending = proposed.directions.filter(({ id }) => !processed.has(id));
    if (pending.length === 0) continue;
    const name = proposed.suggestedName.trim();
    if (!name || pending.some(({ id, text }) => !id || !text.trim())) {
      throw new ProposalAcceptanceError("invalid-input", "A proposed topic and each direction must have nonempty text and identity.");
    }
    let destination = resolveProposedTopicTarget(proposed, topics, acceptance);
    if (!destination) {
      const tag = uniqueTag(slugify(name) || fallbackTag(ordinal), taken);
      taken.add(tag);
      destination = normalizeTopic({
        id: topics.some(({ id }) => id === proposed.id) ? crypto.randomUUID() : proposed.id,
        name, tag, detail: input.detail === true, directions: [],
      });
      topics.push(destination);
      addedTopicCount += 1;
    }
    acceptance.topicTargets[proposed.id] = destination.id;
    const texts = new Set(destination.directions.map(({ text }) => directionTextKey(text)));
    for (const direction of pending) {
      // A preserved candidate identity is an additional replay safeguard if
      // settings were restored independently of the receipt.
      const alreadyPresent = topics.some((topic) => topic.directions.some(({ id }) => id === direction.id));
      const key = directionTextKey(direction.text);
      if (!alreadyPresent && !texts.has(key)) {
        destination.directions.push({ id: direction.id, text: direction.text.trim(), origin: "library" });
        texts.add(key);
        addedDirectionCount += 1;
      }
      processed.add(direction.id);
    }
    const index = topics.findIndex(({ id }) => id === destination.id);
    topics[index] = normalizeTopic(destination);
  }
  acceptance.processedCandidateIds = [...processed].sort();
  return { topics, acceptance, addedTopicCount, addedDirectionCount };
}
