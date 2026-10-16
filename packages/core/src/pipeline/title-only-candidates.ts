import type { TitleOnlyCandidates } from "./pipeline";

/**
 * Read-side guard for persisted or transported counts (run history, CLI run
 * events). The pipeline itself produces `0 <= count <= total`; anything else
 * is ignored by readers rather than displayed or allowed to drop the record.
 */
export function isTitleOnlyCandidates(value: unknown): value is TitleOnlyCandidates {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const keys = Object.keys(value);
  if (keys.length !== 2 || !keys.includes("count") || !keys.includes("total")) return false;
  const { count, total } = value as { count?: unknown; total?: unknown };
  return Number.isSafeInteger(count) && Number.isSafeInteger(total)
    && (count as number) >= 0 && (count as number) <= (total as number);
}
