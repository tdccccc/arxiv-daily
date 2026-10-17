import type { RunState, RunStateEntry, StateStore } from "@arxiv-daily/core";

const RUN_STATUSES = new Set([
  "pending",
  "running",
  "completed",
  "failed_transient",
  "failed_permanent",
  "skipped",
]);

/**
 * Copy run state kept in data.json by old versions into an empty run-state
 * store. Entries the store could not read back are dropped with a warning:
 * written as-is they made every later store write fail, so the plugin could
 * no longer load. A failed migration is reported, never fatal.
 *
 * Returns the legacy state that must stay in data.json: the whole original
 * payload (unreadable entries included) when the copy failed, so the next
 * settings save does not destroy the only copy; empty once it is migrated, or
 * when there is nothing to migrate or the store already has its own state.
 */
export async function migrateLegacyRunState(
  store: StateStore,
  legacy: RunState,
  warn: (message: string) => void,
): Promise<RunState> {
  if (Object.keys(store.snapshot()).length > 0 || Object.keys(legacy).length === 0) return {};
  const readable: RunState = {};
  for (const [date, entry] of Object.entries(legacy)) {
    if (isReadableEntry(entry)) readable[date] = entry;
    else warn(`ignored unreadable legacy run-state entry: ${date}`);
  }
  if (Object.keys(readable).length === 0) return {};
  try {
    await store.replaceAll(readable);
    return {};
  } catch (error) {
    warn(`legacy run state was not migrated: ${error instanceof Error ? error.message : String(error)}`);
    return legacy;
  }
}

function isReadableEntry(entry: unknown): entry is RunStateEntry {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return false;
  const value = entry as Record<string, unknown>;
  return (
    typeof value.status === "string" &&
    RUN_STATUSES.has(value.status) &&
    typeof value.lastAttempt === "number" &&
    typeof value.attempts === "number" &&
    (!("error" in value) || typeof value.error === "string") &&
    (!("papersWritten" in value) ||
      (typeof value.papersWritten === "number" && Number.isFinite(value.papersWritten)))
  );
}
