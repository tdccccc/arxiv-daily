import { createHash } from "node:crypto";
import { seededRandom } from "./export";
import { buildJudgeSystemPrompt, type JudgeContext } from "./summary-judge-contract";
import { SUMMARY_FIELDS, type SummaryFieldKey } from "./summary-checks";
import { requireDirectory, resolveVaultLayout, type VaultLayoutOptions } from "./summary-check";
import { checkNote } from "./summary-note-checks";
import { loadSummaryCorpus, type DetailNote, type SummaryCorpus, type SummaryEntry } from "./summary-corpus";
import { readSourceRecord, rebuildDailyInput, rebuildDetailInput, type SourceRecord } from "./summary-sources";

export const DEFAULT_JUDGE_SEED = "summary-judge";
export const DEFAULT_JUDGE_DAILY = 20;
export const DEFAULT_JUDGE_NOTES = 5;
const SUMMARY_TAG = "summary_under_review";

export interface SampleOptions {
  seed?: string;
  daily?: number;
  notes?: number;
  /** Judge exactly these papers (the daily entry and the note with each id); sampling counts are ignored. */
  ids?: readonly string[];
  /** Sample notes written under earlier prompts too; by default only those that carry every current heading. */
  includeOlderNotes?: boolean;
}

export type SkipReason =
  | "no-source" | "no-input" | "incomplete" | "duplicate-headings" | "not-found" | "fallback"
  | "not-generated" | "error" | "source-changed" | "no-sections";

export interface SkippedItem { kind: "daily" | "note" | "id"; id: string; reason: SkipReason; detail: string }

export interface SampleTargets {
  daily: SummaryEntry[];
  notes: DetailNote[];
  /** Daily entries per report month, every month listed. */
  allocation: Record<string, number>;
  /** Ids asked for that cannot be judged. */
  missing: Array<{ id: string; reason: "not-found" | "fallback" }>;
}

const byDateThenId = (a: { date: string | null; id: string }, b: { date: string | null; id: string }): number =>
  (a.date ?? "").localeCompare(b.date ?? "") || a.id.localeCompare(b.id);

function shuffled<T>(items: readonly T[], seed: string): T[] {
  const result = [...items];
  const random = seededRandom(seed);
  for (let index = 0; index < result.length; index += 1) {
    const pick = index + Math.floor(random() * (result.length - index));
    [result[index], result[pick]] = [result[pick]!, result[index]!];
  }
  return result;
}

/** Largest-remainder shares of `total` over the groups by size; ties go to the earlier group. */
function proportionalShares(sizes: ReadonlyMap<string, number>, total: number): Map<string, number> {
  const all = [...sizes.values()].reduce((sum, size) => sum + size, 0);
  const wanted = Math.min(Math.max(0, total), all);
  const shares = new Map<string, number>();
  const remainders: Array<{ key: string; remainder: number }> = [];
  let given = 0;
  for (const [key, size] of sizes) {
    const share = all === 0 ? 0 : Math.floor((wanted * size) / all);
    shares.set(key, share);
    given += share;
    remainders.push({ key, remainder: all === 0 ? 0 : (wanted * size) % all });
  }
  const order = remainders.map((entry, position) => ({ ...entry, position })).sort((a, b) => b.remainder - a.remainder || a.position - b.position);
  for (const { key } of order.slice(0, wanted - given)) shares.set(key, shares.get(key)! + 1);
  return shares;
}

/** Pick what to judge, deterministically. Pure: no files, no model. */
export function sampleJudgeTargets(corpus: SummaryCorpus, options: SampleOptions = {}): SampleTargets {
  const seed = options.seed ?? DEFAULT_JUDGE_SEED;
  const entries = [...corpus.entries].sort(byDateThenId);
  const months = new Map<string, SummaryEntry[]>();
  for (const entry of entries) months.set(entry.date.slice(0, 7), [...(months.get(entry.date.slice(0, 7)) ?? []), entry]);

  if (options.ids) {
    const wanted = [...new Set(options.ids)];
    const fallbackIds = new Set(corpus.fallbacks.map((fallback) => fallback.id));
    const daily = entries.filter((entry) => wanted.includes(entry.id));
    const notes = corpus.notes.filter((note) => wanted.includes(note.id)).sort((a, b) => a.id.localeCompare(b.id));
    const found = new Set([...daily.map((entry) => entry.id), ...notes.map((note) => note.id)]);
    return {
      daily, notes,
      allocation: Object.fromEntries([...months.keys()].map((month) => [month, daily.filter((entry) => entry.date.startsWith(month)).length])),
      missing: wanted.filter((id) => !found.has(id)).map((id) => ({ id, reason: fallbackIds.has(id) ? "fallback" as const : "not-found" as const })),
    };
  }

  const shares = proportionalShares(new Map([...months].map(([month, group]) => [month, group.length])), options.daily ?? DEFAULT_JUDGE_DAILY);
  const daily = [...months].flatMap(([month, group]) => shuffled(group, `${seed}|${month}`).slice(0, shares.get(month)!)).sort(byDateThenId);
  const eligible = corpus.notes.filter((note) => options.includeOlderNotes || checkNote(note).version === "current").sort((a, b) => a.id.localeCompare(b.id));
  const notes = shuffled(eligible, `${seed}|notes`).slice(0, Math.max(0, options.notes ?? DEFAULT_JUDGE_NOTES)).sort((a, b) => a.id.localeCompare(b.id));
  return { daily, notes, allocation: Object.fromEntries(shares), missing: [] };
}

/** Where the summary under review came from. */
export type SummaryOrigin =
  | { kind: "historical"; vault: string; date: string | null }
  | {
    kind: "regenerated";
    generator: { provider: string; model: string; thinkingMode?: boolean; reasoningEffort?: string };
    generationPromptHash: string;
    generatedAt: string;
    summariesDir: string;
  };

export const hashText = (text: string): string => createHash("sha256").update(text).digest("hex");

export interface JudgeItem {
  kind: "daily" | "note";
  id: string;
  title: string;
  date: string | null;
  /** The report's topic section for a daily entry; the note's primary topic tag for a note. */
  topic: string | null;
  /** Notes only: whether the note carries every heading of the current prompt. */
  version?: "current" | "older";
  /** The parts of the summary a judge may point at, in the order shown, with exactly the text shown. */
  locations: Array<{ key: string; text: string }>;
  /** The whole summary under review as the judge sees it. */
  judgedText: string;
  /** SHA-256 of `judgedText`: a changed summary is judged again. */
  summaryHash: string;
  origin: SummaryOrigin;
  /** The `<paper_data>` message content the generator got. */
  paperData: string;
  messages: Array<{ role: "system" | "user"; content: string }>;
  context: JudgeContext;
  source: { sourcePath: string; arxivVersion: string | null; fetchedAt: string; contentHash: string };
}

export interface BuildItemsOptions {
  sourcesDir: string;
  /** Recorded as the origin of historical summaries. */
  vaultRoot?: string;
  /** Topic tag to display name, as the generator was told it. */
  topics?: ReadonlyArray<{ tag: string; name: string }>;
}

function sourceOf(record: SourceRecord): JudgeItem["source"] {
  return { sourcePath: record.sourcePath, arxivVersion: record.arxivVersion, fetchedAt: record.fetchedAt, contentHash: record.contentHash };
}

function userMessage(paperData: string, judgedText: string): string {
  return `${paperData}\n\n<${SUMMARY_TAG}>\n${judgedText}\n</${SUMMARY_TAG}>`;
}

/** Build the judge messages for the chosen papers; the ones that cannot be built are listed, never replaced. */
export async function buildJudgeItems(
  targets: Pick<SampleTargets, "daily" | "notes">,
  options: BuildItemsOptions,
): Promise<{ items: JudgeItem[]; skipped: SkippedItem[] }> {
  const items: JudgeItem[] = [];
  const skipped: SkippedItem[] = [];
  const skip = (kind: "daily" | "note", id: string, reason: SkipReason, detail: string) => skipped.push({ kind, id, reason, detail });

  for (const entry of targets.daily) {
    const empty = SUMMARY_FIELDS.filter((key: SummaryFieldKey) => !entry.fields[key]);
    if (empty.length > 0) { skip("daily", entry.id, "incomplete", `the entry has no text in ${empty.join(", ")}`); continue; }
    const record = await readSourceRecord(options.sourcesDir, entry.id);
    if (!record) { skip("daily", entry.id, "no-source", "no cached source record"); continue; }
    const rebuilt = await rebuildDailyInput(record, { title: entry.title, authors: entry.authors });
    if (!rebuilt) { skip("daily", entry.id, "no-input", "the cached text is not enough to rebuild the generator's input"); continue; }
    const locations = SUMMARY_FIELDS.map((key) => ({ key, text: entry.fields[key] }));
    const judgedText = locations.map((location) => `[${location.key}] ${location.text}`).join("\n");
    const paperData = rebuilt.messages[1]!.content;
    items.push({
      kind: "daily", id: entry.id, title: entry.title, date: entry.date, topic: entry.topic, locations, judgedText, summaryHash: hashText(judgedText),
      origin: { kind: "historical", vault: options.vaultRoot ?? "", date: entry.date }, paperData,
      messages: [{ role: "system", content: buildJudgeSystemPrompt({ kind: "daily" }) }, { role: "user", content: userMessage(paperData, judgedText) }],
      context: { kind: "daily", locations: Object.fromEntries(locations.map((location) => [location.key, location.text])), sourceText: paperData },
      source: sourceOf(record),
    });
  }

  for (const note of targets.notes) {
    const headings = note.sections.map((section) => section.heading);
    if (new Set(headings).size !== headings.length) { skip("note", note.id, "duplicate-headings", "two sections share a heading, so a location would be ambiguous"); continue; }
    const record = await readSourceRecord(options.sourcesDir, note.id);
    if (!record) { skip("note", note.id, "no-source", "no cached source record"); continue; }
    const topicName = options.topics?.find((topic) => topic.tag === note.topic)?.name ?? note.topic;
    const rebuilt = await rebuildDetailInput(record, { title: note.title, authors: note.authors, topic: note.topic, topics: options.topics });
    if (!rebuilt) { skip("note", note.id, "no-input", "the cached source has no full text, which a detail note is written from"); continue; }
    const locations = note.sections.map((section) => ({ key: section.heading, text: section.body }));
    const judgedText = note.sections.map((section) => `## ${section.heading}\n${section.body}`).join("\n\n");
    const paperData = rebuilt.messages[1]!.content;
    items.push({
      kind: "note", id: note.id, title: note.title, date: note.date, topic: note.topic, version: checkNote(note).version, locations, judgedText, summaryHash: hashText(judgedText),
      origin: { kind: "historical", vault: options.vaultRoot ?? "", date: note.date }, paperData,
      messages: [{ role: "system", content: buildJudgeSystemPrompt({ kind: "note", headings, topic: topicName }) }, { role: "user", content: userMessage(paperData, judgedText) }],
      context: { kind: "note", locations: Object.fromEntries(locations.map((location) => [location.key, location.text])), sourceText: paperData },
      source: sourceOf(record),
    });
  }
  return { items, skipped };
}

export interface SizeStats { items: number; totalChars: number; averageChars: number; maxChars: number }
export interface JudgeEstimate { daily: SizeStats; note: SizeStats; totalChars: number }

/** How much text the judge will be sent: the characters of every message, per kind. */
export function estimateJudgeInput(items: readonly JudgeItem[]): JudgeEstimate {
  const stats = (kind: JudgeItem["kind"]): SizeStats => {
    const sizes = items.filter((item) => item.kind === kind).map((item) => item.messages.reduce((sum, message) => sum + message.content.length, 0));
    const totalChars = sizes.reduce((sum, size) => sum + size, 0);
    return { items: sizes.length, totalChars, averageChars: sizes.length ? Math.round(totalChars / sizes.length) : 0, maxChars: sizes.length ? Math.max(...sizes) : 0 };
  };
  const daily = stats("daily");
  const note = stats("note");
  return { daily, note, totalChars: daily.totalChars + note.totalChars };
}

export interface PrepareJudgeOptions extends VaultLayoutOptions, SampleOptions {
  sourcesDir: string;
  from?: string;
  to?: string;
  topics?: ReadonlyArray<{ tag: string; name: string }>;
}

export interface PreparedJudgeInput {
  corpus: SummaryCorpus;
  sourcesDir: string;
  items: JudgeItem[];
  skipped: SkippedItem[];
  requested: { daily: number; notes: number };
  /** Daily entries sampled per report month. */
  allocation: Record<string, number>;
  /** The ids that were chosen, whether or not a request could be built for them. */
  sample: { daily: string[]; notes: string[] };
  /** How many chosen papers could not be turned into a judge request, per kind. */
  shortfall: { daily: number; notes: number };
  estimate: JudgeEstimate;
}

/** Read the vault and the cached sources, choose what to judge and build the judge requests. No model, no network. */
export async function prepareJudgeInput(options: PrepareJudgeOptions): Promise<PreparedJudgeInput> {
  await requireDirectory(options.sourcesDir, "--sources");
  const { vaultRoot, dailyDir, papersDir, settings } = await resolveVaultLayout(options);
  const corpus = await loadSummaryCorpus({ vaultRoot, dailyDir, papersDir, from: options.from, to: options.to });
  const targets = sampleJudgeTargets(corpus, options);
  const { items, skipped } = await buildJudgeItems(targets, { sourcesDir: options.sourcesDir, vaultRoot, topics: options.topics ?? settings.arxiv.topics });
  const built = (kind: JudgeItem["kind"]) => items.filter((item) => item.kind === kind).length;
  return {
    corpus, sourcesDir: options.sourcesDir, items,
    skipped: [...targets.missing.map((entry) => ({
      kind: "id" as const, id: entry.id, reason: entry.reason,
      detail: entry.reason === "fallback" ? "a fallback entry has no summary to judge" : "no daily entry or detail note with this id",
    })), ...skipped],
    requested: options.ids ? { daily: targets.daily.length, notes: targets.notes.length } : { daily: options.daily ?? DEFAULT_JUDGE_DAILY, notes: options.notes ?? DEFAULT_JUDGE_NOTES },
    allocation: targets.allocation,
    sample: { daily: targets.daily.map((entry) => entry.id), notes: targets.notes.map((note) => note.id) },
    shortfall: { daily: targets.daily.length - built("daily"), notes: targets.notes.length - built("note") },
    estimate: estimateJudgeInput(items),
  };
}
