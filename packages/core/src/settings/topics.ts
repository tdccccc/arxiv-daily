import type { Direction, DirectionOrigin, Topic } from "./types";

const DIRECTION_ORIGINS: readonly DirectionOrigin[] = ["manual", "migrated", "library"];

/**
 * A topic as a template or a caller describes it, before it is given an
 * identity and its directions are settled.
 */
export type TopicSeed = Omit<Topic, "id" | "directions">;

function toOrigin(value: unknown): DirectionOrigin {
  return DIRECTION_ORIGINS.includes(value as DirectionOrigin)
    ? (value as DirectionOrigin)
    : "manual";
}

function toDirection(raw: unknown): Direction | null {
  const stored = (raw ?? {}) as Record<string, unknown>;
  const text = typeof stored.text === "string" ? stored.text.trim() : "";
  if (!text) return null;
  return {
    id: typeof stored.id === "string" && stored.id ? stored.id : crypto.randomUUID(),
    text,
    origin: toOrigin(stored.origin),
  };
}

/**
 * The rollback shadow for a direction list (ADR 0012). The one definition of
 * the rule, so the settings page can keep the shadow in step while editing
 * without becoming a second author of it.
 */
export function deriveTopicDescription(directions: readonly Direction[]): string {
  return directions[0]?.text.trim() ?? "";
}

/**
 * Bring one topic to the current shape and restore the shadow invariant
 * `description === directions[0]?.text ?? ""` (ADR 0012).
 *
 * `directions` is the authority. A topic stored before directions existed
 * carries its interest in `description`, and that text becomes its first
 * direction; once directions are present, a stale `description` written by an
 * older build is overwritten rather than merged. Every path that produces a
 * topic — settings migration, the CLI's TOML reader, the settings page, the
 * quick-start templates — must go through here, so the shadow has exactly one
 * writer.
 */
export function normalizeTopic(raw: unknown): Topic {
  const stored = (raw ?? {}) as Record<string, unknown>;

  // Presence of the key, not its length, decides who is authoritative. A file
  // written before directions existed has no key at all, so its description is
  // migrated. A caller that passes `directions` — the settings page removing
  // the last one included — is stating the full list, and an empty list must
  // stay empty rather than being resurrected from the stale shadow.
  const legacyText = typeof stored.description === "string" ? stored.description.trim() : "";
  const directions: Direction[] = Array.isArray(stored.directions)
    ? stored.directions.map(toDirection).filter((d): d is Direction => d !== null)
    : legacyText
      ? [{ id: crypto.randomUUID(), text: legacyText, origin: "migrated" }]
      : [];

  return {
    id: typeof stored.id === "string" && stored.id ? stored.id : crypto.randomUUID(),
    name: typeof stored.name === "string" ? stored.name : "",
    tag: typeof stored.tag === "string" ? stored.tag : "",
    description: deriveTopicDescription(directions),
    directions,
    detail: stored.detail === true,
  };
}
