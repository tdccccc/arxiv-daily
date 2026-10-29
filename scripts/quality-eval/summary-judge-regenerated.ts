import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { buildCheckpointGenerationIdentity, LLM_TEMPERATURE } from "@arxiv-daily/core";
import { requireDirectory } from "./summary-check";
import { SUMMARY_FIELDS } from "./summary-checks";
import { buildJudgeSystemPrompt } from "./summary-judge-contract";
import { estimateJudgeInput, hashText, type JudgeItem, type PreparedJudgeInput, type SkippedItem } from "./summary-judge-input";
import { parseBody } from "./summary-corpus";
import { readSourceRecord, rebuildDailyInput, rebuildDetailInput } from "./summary-sources";

interface GeneratorIdentity {
  provider: string;
  model: string;
  thinkingMode?: boolean;
  reasoningEffort?: string;
}

interface ResultManifestEntry { kind: "daily" | "note"; id: string; recordHash: string }

interface RegeneratedRecord {
  schemaVersion: number;
  status: "generated" | "fallback" | "error";
  item: {
    kind: "daily" | "note"; id: string; title: string; authors?: string; date: string | null; topic: string | null; version?: "current" | "older";
    source: { contentHash: string };
  };
  generator: GeneratorIdentity;
  summaryLanguage?: "zh" | "en" | null;
  generationPromptHash: string;
  topicName?: { name: string; fromConfig: boolean };
  output: { kind: "structured"; summary: Record<string, string> } | { kind: "markdown"; markdown: string } | { kind: "fallback"; reasonCode: string; attempts: number } | null;
  finishedAt: string;
  error?: string;
}

interface RegenerateRun {
  sample: { daily: string[]; notes: string[] };
  summaryLanguage: "zh" | "en" | null;
  generator: GeneratorIdentity;
  /** Exact generation files committed for this run; absent on historical artifacts. */
  resultManifest?: ResultManifestEntry[];
  skipped?: Array<{ kind: string; id: string }>;
}

export async function readRegenerateRun(summariesDir: string): Promise<RegenerateRun> {
  let run: RegenerateRun;
  try {
    run = JSON.parse(await readFile(path.join(summariesDir, "run.json"), "utf8")) as RegenerateRun;
    if (!run.sample || !Array.isArray(run.sample.daily) || !Array.isArray(run.sample.notes)
      || ![...run.sample.daily, ...run.sample.notes].every((id) => typeof id === "string" && /^\d{4}\.\d{4,5}$/.test(id))) throw new Error("no sample");
  } catch {
    throw new Error(`${path.join(summariesDir, "run.json")} is missing or is not the run.json of summary-regenerate.mjs; --summaries must name its --out directory`);
  }
  if (Object.hasOwn(run, "resultManifest")) {
    if (!Array.isArray(run.resultManifest)) throw new Error("The generation resultManifest must be an array; it cannot be read as a legacy run.");
    const seen = new Set<string>();
    for (const entry of run.resultManifest) {
      if (!entry || (entry.kind !== "daily" && entry.kind !== "note") || typeof entry.id !== "string"
        || !run.sample[entry.kind === "daily" ? "daily" : "notes"].includes(entry.id)
        || typeof entry.recordHash !== "string" || !/^[0-9a-f]{64}$/.test(entry.recordHash)
        || seen.has(`${entry.kind}:${entry.id}`)) {
        throw new Error("The generation resultManifest has a duplicate, invalid or out-of-sample entry.");
      }
      seen.add(`${entry.kind}:${entry.id}`);
    }
  }
  return run;
}

async function readRecord(summariesDir: string, kind: "daily" | "note", id: string, recordHash?: string): Promise<RegeneratedRecord | null> {
  const file = path.join(summariesDir, "summaries", `${kind}-${id}.json`);
  if (!existsSync(file)) return null;
  const text = await readFile(file, "utf8");
  if (recordHash !== undefined && hashText(text) !== recordHash) {
    throw new Error(`Generation resultManifest hash mismatch for ${kind} ${id}; the summary is not the result committed by this run.`);
  }
  try {
    const record = JSON.parse(text) as RegeneratedRecord;
    return record.schemaVersion === 1 && record.item?.id === id && record.item.kind === kind ? record : null;
  } catch {
    return null;
  }
}

/** Compare the public model and mode fields retained by legacy outputs. */
function comparableGenerator(generator: GeneratorIdentity, kind: "daily" | "note"): string | null {
  if (!generator || typeof generator.provider !== "string" || typeof generator.model !== "string"
    || (generator.thinkingMode !== undefined && typeof generator.thinkingMode !== "boolean")
    || (generator.reasoningEffort !== undefined && typeof generator.reasoningEffort !== "string")) return null;
  const mode = generator.thinkingMode === undefined
    ? { kind: "unknown", reasoningEffort: generator.reasoningEffort ?? null }
    : buildCheckpointGenerationIdentity({
      ...generator,
      // Only the normalized mode is used below; legacy artifacts intentionally omit endpoints.
      baseUrl: "https://evaluation.invalid/v1",
      thinkingMode: generator.thinkingMode,
      reasoningEffort: generator.provider === "zhipu" ? "" : generator.reasoningEffort ?? "",
    }, kind === "daily" ? 0 : LLM_TEMPERATURE).mode;
  return JSON.stringify({ provider: generator.provider, model: generator.model, mode });
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
    const manifest = run.resultManifest?.find((entry) => entry.kind === kind && entry.id === id);
    if (run.resultManifest && !manifest) { skip("not-generated", "this run has not committed a regenerated result for this paper"); continue; }
    if (run.skipped?.some((entry) => entry.id === id && (entry.kind === kind || entry.kind === "id"))) {
      skip("not-generated", "the regenerate run skipped this paper; any older result is not part of the run");
      continue;
    }
    const record = await readRecord(summariesDir, kind, id, manifest?.recordHash);
    if (!record) { skip("not-generated", "no regenerated summary was written for this paper"); continue; }
    const generator = comparableGenerator(record.generator, kind);
    if (generator === null || generator !== comparableGenerator(run.generator, kind)) {
      skip("not-generated", "the summary's generator does not match the generator of this run");
      continue;
    }
    if (record.summaryLanguage !== undefined && (record.summaryLanguage ?? "zh") !== (run.summaryLanguage ?? "zh")) {
      skip("not-generated", "the summary language does not match the language of this run");
      continue;
    }
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
