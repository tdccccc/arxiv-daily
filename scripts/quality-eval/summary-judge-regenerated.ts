import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { requireDirectory } from "./summary-check";
import { SUMMARY_FIELDS } from "./summary-checks";
import { buildJudgeSystemPrompt } from "./summary-judge-contract";
import { estimateJudgeInput, hashText, type JudgeItem, type PreparedJudgeInput, type SkippedItem } from "./summary-judge-input";
import { parseBody } from "./summary-corpus";
import { readSourceRecord, rebuildDailyInput, rebuildDetailInput } from "./summary-sources";

interface RegeneratedRecord {
  schemaVersion: number;
  status: "generated" | "fallback" | "error";
  item: {
    kind: "daily" | "note"; id: string; title: string; authors?: string; date: string | null; topic: string | null; version?: "current" | "older";
    source: { contentHash: string };
  };
  generator: { provider: string; model: string; thinkingMode?: boolean; reasoningEffort?: string };
  generationPromptHash: string;
  topicName?: { name: string; fromConfig: boolean };
  output: { kind: "structured"; summary: Record<string, string> } | { kind: "markdown"; markdown: string } | { kind: "fallback"; reasonCode: string; attempts: number } | null;
  finishedAt: string;
  error?: string;
}

interface RegenerateRun {
  sample: { daily: string[]; notes: string[] };
  summaryLanguage: "zh" | "en" | null;
  generator: { provider: string; model: string; thinkingMode?: boolean; reasoningEffort?: string };
}

export async function readRegenerateRun(summariesDir: string): Promise<RegenerateRun> {
  try {
    const run = JSON.parse(await readFile(path.join(summariesDir, "run.json"), "utf8")) as RegenerateRun;
    if (!run.sample || !Array.isArray(run.sample.daily) || !Array.isArray(run.sample.notes)) throw new Error("no sample");
    return run;
  } catch {
    throw new Error(`${path.join(summariesDir, "run.json")} is missing or is not the run.json of summary-regenerate.mjs; --summaries must name its --out directory`);
  }
}

async function readRecord(summariesDir: string, kind: "daily" | "note", id: string): Promise<RegeneratedRecord | null> {
  const file = path.join(summariesDir, "summaries", `${kind}-${id}.json`);
  if (!existsSync(file)) return null;
  try {
    const record = JSON.parse(await readFile(file, "utf8")) as RegeneratedRecord;
    return record.schemaVersion === 1 && record.item?.id === id ? record : null;
  } catch {
    return null;
  }
}

export interface PrepareRegeneratedOptions {
  /** The `--out` directory of summary-regenerate.mjs. */
  summariesDir: string;
  sourcesDir: string;
}

/**
 * Build the judge's requests from regenerated summaries instead of the vault's. The sample is the one the
 * regenerate run recorded. The input is rebuilt from the cached source record exactly as the generator got it, and
 * only when that record still has the content hash the summary was generated from.
 */
export async function prepareRegeneratedJudgeInput(options: PrepareRegeneratedOptions): Promise<PreparedJudgeInput & { generator: RegenerateRun["generator"]; summaryLanguage: RegenerateRun["summaryLanguage"] }> {
  await requireDirectory(options.sourcesDir, "--sources");
  const summariesDir = path.resolve(options.summariesDir);
  const run = await readRegenerateRun(summariesDir);
  const language = run.summaryLanguage ?? undefined;
  const items: JudgeItem[] = [];
  const skipped: SkippedItem[] = [];

  const wanted = [
    ...run.sample.daily.map((id) => ({ kind: "daily" as const, id })),
    ...run.sample.notes.map((id) => ({ kind: "note" as const, id })),
  ];
  for (const { kind, id } of wanted) {
    const skip = (reason: SkippedItem["reason"], detail: string) => skipped.push({ kind, id, reason, detail });
    const record = await readRecord(summariesDir, kind, id);
    if (!record) { skip("not-generated", "no regenerated summary was written for this paper"); continue; }
    if (record.status === "fallback") {
      const code = record.output && record.output.kind === "fallback" ? record.output.reasonCode : "unknown";
      skip("fallback", `the generator fell back (${code}), so there is no summary to judge`);
      continue;
    }
    if (record.status === "error" || !record.output) { skip("error", `the generation failed: ${record.error ?? "no output"}`); continue; }
    const source = await readSourceRecord(options.sourcesDir, id);
    if (!source) { skip("no-source", "no cached source record"); continue; }
    if (source.contentHash !== record.item.source.contentHash) { skip("source-changed", "the cached source text is not the text this summary was generated from"); continue; }
    if (record.item.authors === undefined) { skip("no-input", "the regenerated record does not name the authors the generator was given, so its input cannot be rebuilt"); continue; }

    let locations: Array<{ key: string; text: string }>;
    let judgedText: string;
    let systemPrompt: string;
    let rebuilt: { messages: Array<{ role: string; content: string }> } | null;
    if (kind === "daily") {
      if (record.output.kind !== "structured") { skip("error", "the record holds no structured summary"); continue; }
      const summary = record.output.summary;
      locations = SUMMARY_FIELDS.map((key) => ({ key, text: String(summary[key] ?? "") }));
      if (locations.some((location) => !location.text)) { skip("incomplete", "the regenerated summary has an empty field"); continue; }
      judgedText = locations.map((location) => `[${location.key}] ${location.text}`).join("\n");
      rebuilt = await rebuildDailyInput(source, { title: record.item.title, authors: record.item.authors, summaryLanguage: language });
      systemPrompt = buildJudgeSystemPrompt({ kind: "daily" });
    } else {
      if (record.output.kind !== "markdown") { skip("error", "the record holds no Markdown note"); continue; }
      const sections = parseBody(record.output.markdown).sections;
      const headings = sections.map((section) => section.heading);
      if (headings.length === 0) { skip("no-sections", "the regenerated note has no ## sections"); continue; }
      if (new Set(headings).size !== headings.length) { skip("duplicate-headings", "two sections share a heading, so a location would be ambiguous"); continue; }
      locations = sections.map((section) => ({ key: section.heading, text: section.body }));
      judgedText = sections.map((section) => `## ${section.heading}\n${section.body}`).join("\n\n");
      const topicName = record.topicName?.name ?? record.item.topic ?? "";
      rebuilt = await rebuildDetailInput(source, { title: record.item.title, authors: record.item.authors, topic: record.item.topic ?? "", topics: [{ tag: record.item.topic ?? "", name: topicName }] as never, summaryLanguage: language });
      systemPrompt = buildJudgeSystemPrompt({ kind: "note", headings, topic: topicName });
    }
    if (!rebuilt) { skip("no-input", "the cached text cannot rebuild the generator's input"); continue; }
    const paperData = rebuilt.messages[1]!.content;
    items.push({
      kind, id, title: record.item.title, date: record.item.date, topic: record.item.topic,
      ...(kind === "note" && record.item.version ? { version: record.item.version } : {}),
      locations, judgedText, summaryHash: hashText(judgedText),
      origin: { kind: "regenerated", generator: record.generator, generationPromptHash: record.generationPromptHash, generatedAt: record.finishedAt, summariesDir },
      paperData,
      messages: [{ role: "system", content: systemPrompt }, { role: "user", content: `${paperData}\n\n<summary_under_review>\n${judgedText}\n</summary_under_review>` }],
      context: { kind, locations: Object.fromEntries(locations.map((location) => [location.key, location.text])), sourceText: paperData },
      source: { sourcePath: source.sourcePath, arxivVersion: source.arxivVersion, fetchedAt: source.fetchedAt, contentHash: source.contentHash },
    });
  }
  const built = (kind: JudgeItem["kind"]) => items.filter((item) => item.kind === kind).length;
  return {
    corpus: undefined as never, sourcesDir: options.sourcesDir, items, skipped,
    requested: { daily: run.sample.daily.length, notes: run.sample.notes.length },
    allocation: (run as unknown as { allocation?: Record<string, number> }).allocation ?? {},
    sample: run.sample,
    shortfall: { daily: run.sample.daily.length - built("daily"), notes: run.sample.notes.length - built("note") },
    estimate: estimateJudgeInput(items),
    generator: run.generator, summaryLanguage: run.summaryLanguage,
  };
}
