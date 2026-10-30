import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

export interface SummaryResultManifestEntry {
  kind: "daily" | "note";
  id: string;
  recordHash: string;
}

export const summaryResultKey = (kind: string, id: string): string => `${kind}-${id}`;
export const isSummaryResultKind = (value: unknown): value is "daily" | "note" => value === "daily" || value === "note";
export const isSummaryResultId = (value: unknown): value is string => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value);

/** Read the exact files admitted by a run, without treating directory leftovers as results. */
export async function readSummaryResultManifest<T>(
  directory: string,
  manifest: unknown,
  sample: ReadonlySet<string>,
  skipped: ReadonlySet<string>,
  accept: (value: unknown, entry: SummaryResultManifestEntry) => value is T,
): Promise<T[]> {
  const fail = (detail: string) => new Error(`invalid result manifest in ${directory}: ${detail}; rerun the evaluation to recover the current results`);
  if (!Array.isArray(manifest)) throw fail("expected an array");
  const seen = new Set<string>();
  const records: T[] = [];
  for (const entry of manifest as SummaryResultManifestEntry[]) {
    if (!entry || !isSummaryResultKind(entry.kind) || !isSummaryResultId(entry.id) || !/^[a-f0-9]{64}$/.test(entry.recordHash ?? "")) throw fail("malformed entry");
    const key = summaryResultKey(entry.kind, entry.id);
    if (seen.has(key) || !sample.has(key) || skipped.has(key)) throw fail(`${key} is duplicate, outside the sample, or skipped`);
    seen.add(key);
    let content: Buffer;
    try { content = await readFile(path.join(directory, `${key}.json`)); }
    catch { throw fail(`${key} is missing or unreadable`); }
    if (createHash("sha256").update(content).digest("hex") !== entry.recordHash) throw fail(`${key} no longer matches its recorded hash`);
    let record: unknown;
    try { record = JSON.parse(content.toString("utf8")); }
    catch { throw fail(`${key} is not valid JSON`); }
    if (!accept(record, entry)) throw fail(`${key} has a different item or model identity, or is malformed`);
    records.push(record);
  }
  return records;
}
