export const DEFAULT_MAX_DAILY_PAPERS = 20;

export function isValidMaxDailyPapers(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

/** Missing or invalid persisted limits use the shared daily default. */
export function normalizeMaxDailyPapers(value: unknown): number {
  return isValidMaxDailyPapers(value) ? value : DEFAULT_MAX_DAILY_PAPERS;
}
