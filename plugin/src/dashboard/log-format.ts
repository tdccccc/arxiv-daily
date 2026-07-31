export const DEFAULT_LOG_LEVELS: ReadonlySet<string> = new Set(["debug", "info", "warn", "error"]);

const LOG_LEVEL_TAG = /\[(DEBUG|INFO|WARN|ERROR)\]/;

function parseLogLevelTag(line: string): string | null {
  const m = line.match(LOG_LEVEL_TAG);
  return m?.[1] ? m[1].toLowerCase() : null;
}

export interface FormatLogEntriesOptions {
  /** Levels to keep. Defaults to all four levels. */
  levels?: Set<string>;
}

export function formatLogEntries(
  buffer: string[],
  opts: FormatLogEntriesOptions = {},
): string {
  if (buffer.length === 0) return "(no log entries)";
  const levels = opts.levels ?? DEFAULT_LOG_LEVELS;
  const kept: string[] = [];
  // Iterate oldest→newest, push allowed; untagged lines are kept.
  for (const line of buffer) {
    const lvl = parseLogLevelTag(line);
    if (lvl === null || levels.has(lvl)) kept.push(line);
  }
  if (kept.length === 0) return "(no log entries at this level)";
  // Reverse so newest is on top.
  return kept.reverse().join("\n");
}
