/**
 * Which directions of a topic selected a paper (ADR 0012 §1).
 *
 * Deliberately not folded into `PaperDiscoveryProvenance`: that structure
 * requires every direction to name at least one representative paper as
 * evidence, and an accepted topic direction keeps no evidence at all (ADR 0012
 * §4). Sharing one validator would mean loosening that requirement for both.
 */

export const TOPIC_DIRECTION_HIT_MAX_COUNT = 256 as const;
export const TOPIC_DIRECTION_HIT_MAX_TAG_LENGTH = 120 as const;
export const TOPIC_DIRECTION_HIT_MAX_ID_LENGTH = 128 as const;
export const TOPIC_DIRECTION_HIT_MAX_TEXT_LENGTH = 1_000 as const;

export interface TopicDirectionHit {
  /** Tag of the topic the paper was filed under. */
  tag: string;
  /** The direction's stored identity, stable across runs. */
  id: string;
  /** The direction line as the researcher wrote it. Untrusted display text. */
  text: string;
}

/**
 * Validate a hit list, or return null. Hits keep the topic's own order — the
 * order the researcher sees in settings — so they are never re-sorted here.
 */
export function normalizeTopicDirectionHits(value: unknown): TopicDirectionHit[] | null {
  if (!Array.isArray(value) || value.length === 0
    || value.length > TOPIC_DIRECTION_HIT_MAX_COUNT) return null;

  const hits: TopicDirectionHit[] = [];
  const seen = new Set<string>();
  let firstTag: string | undefined;
  for (const raw of value) {
    if (!isExactDataObject(raw, ["tag", "id", "text"])
      || !isBoundedText(raw.tag, TOPIC_DIRECTION_HIT_MAX_TAG_LENGTH)
      || !isBoundedText(raw.id, TOPIC_DIRECTION_HIT_MAX_ID_LENGTH)
      || !isBoundedText(raw.text, TOPIC_DIRECTION_HIT_MAX_TEXT_LENGTH)) return null;
    // A paper is filed under exactly one topic, so every hit on it belongs to
    // that same topic. A mixed list means the hits were assembled wrongly.
    if (firstTag === undefined) firstTag = raw.tag;
    else if (raw.tag !== firstTag) return null;
    if (seen.has(raw.id)) return null;
    seen.add(raw.id);
    hits.push({ tag: raw.tag, id: raw.id, text: raw.text });
  }
  return hits;
}

function isExactDataObject(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype: unknown = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) return false;
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length
    && actual.every((key, index) => key === expected[index])
    && actual.every((key) => {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      return Boolean(descriptor && "value" in descriptor);
    });
}

function isBoundedText(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum;
}
