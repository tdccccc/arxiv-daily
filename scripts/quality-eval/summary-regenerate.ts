import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  describeValidationFailureForLog, LlmClient, Logger, summarizeDailyPaperWithValidationRetry, summarizePaperDetail,
  type LlmCallMetrics, type LlmSettings, type StructuredPaperSummary, type SummaryLanguage,
} from "@arxiv-daily/core";
import { NodeHttpClient } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../../apps/cli/src/config";
import { prepareOutDir } from "./evaluate";
import { redactor } from "./rerun";
import { requireDate, requireDirectory, resolveVaultLayout, type VaultLayoutOptions } from "./summary-check";
import { loadSummaryCorpus, type DetailNote, type SummaryEntry } from "./summary-corpus";
import { sampleJudgeTargets, type SampleOptions, type SkippedItem } from "./summary-judge-input";
import { checkNote } from "./summary-note-checks";
import { CHARS_PER_TOKEN } from "./summary-judge-run";
import {
  dailyPaperFromRecord, detailCallFromRecord, readSourceRecord, rebuildDailyInput, rebuildDetailInput, type SourceRecord,
} from "./summary-sources";

const MAX_CONSECUTIVE_ERRORS = 3;
/** The pipeline's validation-retry budget for one daily paper (core's DAILY_PAPER_SUMMARY_MAX_ATTEMPTS). */
const DAILY_ATTEMPTS = 3;
const SHOWN_SKIPPED = 10;

export const sha256Hex = (text: string): string => createHash("sha256").update(text).digest("hex");

export interface SummaryRegenerateOptions extends VaultLayoutOptions, SampleOptions {
  sourcesDir: string;
  outDir: string;
  from?: string;
  to?: string;
  yes?: boolean;
  /** Generate again even when a result for the same model, prompt and source text exists. */
  regenerate?: boolean;
}

export interface GeneratorConfig {
  llm: LlmSettings;
  topics: Array<{ tag: string; name: string }>;
  summaryLanguage?: SummaryLanguage;
}

export interface GeneratorModelLike {
  call(
    messages: Array<{ role: "system" | "user" | "assistant"; content: string }>,
    options?: { temperature?: number; onMetrics?: (metrics: LlmCallMetrics) => void },
  ): Promise<string>;
}

export interface SummaryRegenerateDeps {
  /** The generator client; tests pass a fake, otherwise one is built from the CLI config's `[llm]`. */
  llm?: GeneratorModelLike;
  now?: () => Date;
  log?: (line: string) => void;
  loadGeneratorConfig?: (configPath?: string) => Promise<GeneratorConfig>;
}

export interface RegeneratePlan {
  items: { daily: number; note: number };
  skipped: SkippedItem[];
  alreadyDone: number;
  toGenerate: number;
  maxRequests: number;
  totalChars: number;
  estimatedTokens: number;
}

export interface RegenerateCounts { generated: number; fallback: number; errors: number; reused: number; skipped: number; remaining: number }

export interface SummaryRegenerateResult {
  dryRun: boolean;
  outDir: string;
  plan: RegeneratePlan;
  counts: RegenerateCounts | null;
  stopped: "errors" | null;
}

export function parseSummaryRegenerateArgs(argv: readonly string[]): Partial<SummaryRegenerateOptions> & { help?: boolean } {
  const parsed: Partial<SummaryRegenerateOptions> & { help?: boolean } = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]!;
    if (flag === "--help" || flag === "-h") return { help: true };
    if (flag === "--yes") { parsed.yes = true; continue; }
    if (flag === "--regenerate") { parsed.regenerate = true; continue; }
    if (flag === "--include-older-notes") { parsed.includeOlderNotes = true; continue; }
    const take = (): string => {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) throw new Error(`${flag} needs a value`);
      index += 1;
      return value;
    };
    const count = (): number => {
      const value = take();
      if (!/^\d+$/.test(value)) throw new Error(`${flag} must be a whole number, 0 or more`);
      return Number(value);
    };
    switch (flag) {
      case "--config": parsed.configPath = take(); break;
      case "--vault": parsed.vaultRoot = take(); break;
      case "--daily-dir": parsed.dailyDir = take(); break;
      case "--papers-dir": parsed.papersDir = take(); break;
      case "--sources": parsed.sourcesDir = take(); break;
      case "--out": parsed.outDir = take(); break;
      case "--seed": parsed.seed = take(); break;
      case "--daily": parsed.daily = count(); break;
      case "--notes": parsed.notes = count(); break;
      case "--from": parsed.from = requireDate(flag, take()); break;
      case "--to": parsed.to = requireDate(flag, take()); break;
      case "--ids": {
        const ids = take().split(",").map((id) => id.trim()).filter(Boolean);
        if (ids.length === 0) throw new Error("--ids needs at least one arXiv id, separated by commas");
        parsed.ids = ids;
        break;
      }
      default: throw new Error(`unknown option: ${flag}`);
    }
  }
  if (!parsed.sourcesDir) throw new Error("--sources <directory> is required");
  if (!parsed.outDir) throw new Error("--out <directory> is required");
  if (parsed.from && parsed.to && parsed.from > parsed.to) {
    throw new Error(`--from ${parsed.from} is after --to ${parsed.to}; the start date is after the end date`);
  }
  return parsed;
}

async function defaultLoadGeneratorConfig(configPath?: string): Promise<GeneratorConfig> {
  const config = await loadCliConfig({ configPath });
  return {
    llm: config.settings.llm,
    topics: config.settings.arxiv.topics.map((topic) => ({ tag: topic.tag, name: topic.name })),
    summaryLanguage: config.settings.output.summaryLanguage,
  };
}

type Messages = Array<{ role: string; content: string }>;

interface Job {
  kind: "daily" | "note";
  id: string;
  title: string;
  date: string | null;
  topic: string | null;
  version?: "current" | "older";
  topicName?: { name: string; fromConfig: boolean };
  record: SourceRecord;
  entry?: SummaryEntry;
  note?: DetailNote;
  /** The system and user messages of the first request, as the pipeline's functions build them. */
  messages: Messages;
  promptHash: string;
}

const fileFor = (outDir: string, job: Pick<Job, "kind" | "id">): string => path.join(outDir, "summaries", `${job.kind}-${job.id}.json`);

async function writeAtomic(file: string, text: string): Promise<void> {
  const temporary = `${file}.tmp`;
  await writeFile(temporary, text);
  await rename(temporary, file);
}

async function isCurrent(file: string, job: Job, model: LlmSettings): Promise<boolean> {
  try {
    const record = JSON.parse(await readFile(file, "utf8")) as {
      status: string; generator: { provider: string; model: string }; generationPromptHash: string; item: { source: { contentHash: string } };
    };
    return (record.status === "generated" || record.status === "fallback")
      && record.generator.model === model.model && record.generator.provider === model.provider
      && record.generationPromptHash === job.promptHash && record.item.source.contentHash === job.record.contentHash;
  } catch {
    return false;
  }
}

/** One model call's usage, summed over the calls a paper took. */
function addUsage(total: Usage, metrics: LlmCallMetrics): void {
  total.calls += 1;
  total.inputTokens += metrics.inputTokens ?? 0;
  total.outputTokens += metrics.outputTokens ?? 0;
  total.totalTokens += metrics.totalTokens ?? 0;
  total.usageComplete &&= metrics.usageComplete;
}
interface Usage { calls: number; inputTokens: number; outputTokens: number; totalTokens: number; usageComplete: boolean }

/**
 * Write the sampled papers' summaries again, with the current generator and prompts, on the cached source text.
 * Daily entries go through the pipeline's own validation-retry function and its fallback classification; notes go
 * through `summarizePaperDetail`. Without `yes` it only prints the plan. Each finished paper is written to
 * `<out>/summaries/` as it completes, so an interrupted run resumes where it stopped.
 */
export async function runSummaryRegenerate(options: SummaryRegenerateOptions, deps: SummaryRegenerateDeps = {}): Promise<SummaryRegenerateResult> {
  const log = deps.log ?? (() => undefined);
  const now = deps.now ?? (() => new Date());
  const outDir = path.resolve(options.outDir);

  const generator = await (deps.loadGeneratorConfig ?? defaultLoadGeneratorConfig)(options.configPath);
  await requireDirectory(options.sourcesDir, "--sources");
  const { vaultRoot, dailyDir, papersDir } = await resolveVaultLayout(options);
  const corpus = await loadSummaryCorpus({ vaultRoot, dailyDir, papersDir, from: options.from, to: options.to });
  const targets = sampleJudgeTargets(corpus, options);
  const language = generator.summaryLanguage;

  const skipped: SkippedItem[] = targets.missing.map((entry) => ({
    kind: "id" as const, id: entry.id, reason: entry.reason,
    detail: entry.reason === "fallback" ? "a fallback entry has no summary to regenerate" : "no daily entry or detail note with this id",
  }));
  const jobs: Job[] = [];
  for (const entry of targets.daily) {
    const record = await readSourceRecord(options.sourcesDir, entry.id);
    if (!record) { skipped.push({ kind: "daily", id: entry.id, reason: "no-source", detail: "no cached source record" }); continue; }
    const rebuilt = await rebuildDailyInput(record, { title: entry.title, authors: entry.authors, summaryLanguage: language });
    if (!rebuilt) { skipped.push({ kind: "daily", id: entry.id, reason: "no-input", detail: "the cached text is not enough for the pipeline to summarize from (it would fall back without a model call)" }); continue; }
    jobs.push({ kind: "daily", id: entry.id, title: entry.title, date: entry.date, topic: entry.topic, record, entry, messages: rebuilt.messages, promptHash: sha256Hex(rebuilt.messages[0]!.content) });
  }
  for (const note of targets.notes) {
    const record = await readSourceRecord(options.sourcesDir, note.id);
    if (!record) { skipped.push({ kind: "note", id: note.id, reason: "no-source", detail: "no cached source record" }); continue; }
    const configured = generator.topics.find((topic) => topic.tag === note.topic);
    const rebuilt = await rebuildDetailInput(record, { title: note.title, authors: note.authors, topic: note.topic, topics: generator.topics as never, summaryLanguage: language });
    if (!rebuilt) { skipped.push({ kind: "note", id: note.id, reason: "no-input", detail: "the cached source has no full text, which a detail note is written from" }); continue; }
    jobs.push({
      kind: "note", id: note.id, title: note.title, date: note.date, topic: note.topic, version: checkNote(note).version,
      topicName: { name: configured?.name || note.topic, fromConfig: Boolean(configured?.name) }, record, note,
      messages: rebuilt.messages, promptHash: sha256Hex(rebuilt.messages[0]!.content),
    });
  }

  const pending: Job[] = [];
  let reused = 0;
  for (const job of jobs) {
    if (!options.regenerate && await isCurrent(fileFor(outDir, job), job, generator.llm)) reused += 1;
    else pending.push(job);
  }
  const chars = (job: Job): number => job.messages.reduce((sum, message) => sum + message.content.length, 0);
  const totalChars = pending.reduce((sum, job) => sum + chars(job), 0);
  const dailyPending = pending.filter((job) => job.kind === "daily").length;
  const plan: RegeneratePlan = {
    items: { daily: jobs.filter((job) => job.kind === "daily").length, note: jobs.filter((job) => job.kind === "note").length },
    skipped, alreadyDone: reused, toGenerate: pending.length,
    maxRequests: dailyPending * DAILY_ATTEMPTS + (pending.length - dailyPending), totalChars, estimatedTokens: Math.ceil(totalChars / CHARS_PER_TOKEN),
  };

  log(`Generator model: ${generator.llm.model} (${generator.llm.provider}).`);
  log(`Items: daily ${plan.items.daily}, note ${plan.items.note} (${jobs.length} items). Already generated with this model, prompt and source: ${reused}. To generate: ${pending.length}.`);
  if (skipped.length > 0) log(`Skipped ${skipped.length} (not replaced): ${skipped.slice(0, SHOWN_SKIPPED).map((entry) => `${entry.id} [${entry.reason}]`).join(", ")}${skipped.length > SHOWN_SKIPPED ? ", ..." : ""}`);
  log(`This ${options.yes ? "will" : "would"} make at most ${plan.maxRequests} requests: a daily entry gets up to ${DAILY_ATTEMPTS} attempts (the pipeline retries a reply that fails validation, ${dailyPending} x ${DAILY_ATTEMPTS}), a note one request (${pending.length - dailyPending}). The model client may also retry transient HTTP failures inside a request. First requests carry ${totalChars} characters, about ${plan.estimatedTokens} input tokens (estimated at ${CHARS_PER_TOKEN} characters per token for mixed English and Chinese text).`);
  if (!options.yes) {
    log("No model was called and nothing was written. Add --yes to generate.");
    return { dryRun: true, outDir, plan, counts: null, stopped: null };
  }

  await prepareOutDir(outDir);
  await mkdir(path.join(outDir, "summaries"), { recursive: true });
  const model = generator.llm;
  const redact = redactor(model);
  const client: GeneratorModelLike = deps.llm ?? (() => {
    const logger = new Logger("error");
    logger.setSensitiveValues([model.apiKey]);
    return new LlmClient(model, logger, new NodeHttpClient());
  })();
  const identity = { provider: model.provider, model: model.model, thinkingMode: model.thinkingMode, reasoningEffort: model.reasoningEffort };

  const counts: RegenerateCounts = { generated: 0, fallback: 0, errors: 0, reused, skipped: skipped.length, remaining: 0 };
  const runFile = path.join(outDir, "run.json");
  const run = {
    schemaVersion: 1, startedAt: now().toISOString(), finishedAt: null as string | null,
    // The CLI config path and the secrets in it are not recorded.
    args: { daily: options.daily ?? null, notes: options.notes ?? null, ids: options.ids ?? null, from: options.from ?? null, to: options.to ?? null, includeOlderNotes: Boolean(options.includeOlderNotes), regenerate: Boolean(options.regenerate), sourcesDir: path.resolve(options.sourcesDir) },
    seed: options.seed ?? "summary-judge", allocation: targets.allocation,
    generator: identity, summaryLanguage: language ?? null,
    sample: { daily: targets.daily.map((entry) => entry.id), notes: targets.notes.map((note) => note.id) },
    skipped, plan, counts: null as RegenerateCounts | null, stopped: null as "errors" | null,
  };
  await writeAtomic(runFile, `${JSON.stringify(run, null, 2)}\n`);

  let consecutiveErrors = 0;
  let stopped: "errors" | null = null;
  let attempted = 0;
  for (const job of pending) {
    if (stopped) break;
    attempted += 1;
    const started = now();
    const usage: Usage = { calls: 0, inputTokens: 0, outputTokens: 0, totalTokens: 0, usageComplete: true };
    const failures: string[] = [];
    const onMetrics = (metrics: LlmCallMetrics): void => addUsage(usage, metrics);
    let output: Record<string, unknown> | null = null;
    let status: "generated" | "fallback" | "error" = "error";
    let error: string | undefined;
    let attempts = 0;
    // Counts requests, whatever the wrapped function does with the reply.
    const counting = { call: (messages: Parameters<GeneratorModelLike["call"]>[0], options?: Parameters<GeneratorModelLike["call"]>[1]) => { attempts += 1; return client.call(messages, options); } };
    try {
      if (job.kind === "daily") {
        const identityInput = { title: job.title, authors: job.entry!.authors, summaryLanguage: language };
        const result = await summarizeDailyPaperWithValidationRetry(dailyPaperFromRecord(job.record, identityInput), {
          llm: counting as never, summaryLanguage: language, onMetrics,
          onValidationFailure: (failure) => { failures.push(describeValidationFailureForLog(failure)); },
        });
        if (result.kind === "structured") {
          status = "generated";
          output = { kind: "structured", summary: result.summary as StructuredPaperSummary };
        } else if (result.reasonCode === "transport-exhausted") {
          // The pipeline sets such a paper aside for a retry; a result file would freeze the outage as an outcome.
          status = "error";
          error = `transport-exhausted after ${result.attempts} attempt${result.attempts === 1 ? "" : "s"}: the model endpoint could not be reached`;
        } else {
          status = "fallback";
          output = { kind: "fallback", reasonCode: result.reasonCode, attempts: result.attempts };
        }
      } else {
        const { paper, deps: detailDeps } = detailCallFromRecord(job.record, { title: job.title, authors: job.note!.authors, topic: job.topic!, topics: generator.topics as never, summaryLanguage: language });
        const markdown = await summarizePaperDetail(paper, { ...detailDeps, llm: counting as never, onMetrics });
        status = "generated";
        output = { kind: "markdown", markdown };
      }
    } catch (caught) {
      status = "error";
      error = redact((caught as Error)?.message ?? caught) || "unknown error";
    }

    const finishedAt = now();
    const record = {
      schemaVersion: 1, status,
      item: { kind: job.kind, id: job.id, title: job.title, authors: job.kind === "daily" ? job.entry!.authors : job.note!.authors, date: job.date, topic: job.topic, ...(job.version ? { version: job.version } : {}), source: { sourcePath: job.record.sourcePath, arxivVersion: job.record.arxivVersion, fetchedAt: job.record.fetchedAt, contentHash: job.record.contentHash } },
      generator: identity, generationPromptHash: job.promptHash,
      ...(job.topicName ? { topicName: job.topicName } : {}),
      output, attempts, usage, validationFailures: failures.map((failure) => redact(failure)),
      ...(error !== undefined ? { error } : {}),
      startedAt: started.toISOString(), finishedAt: finishedAt.toISOString(), elapsedMs: finishedAt.getTime() - started.getTime(),
    };
    await writeAtomic(fileFor(outDir, job), `${JSON.stringify(record, null, 2)}\n`);

    if (status === "error") {
      counts.errors += 1;
      consecutiveErrors += 1;
      log(`[${attempted}/${pending.length}] ${job.kind} ${job.id}: error: ${error}`);
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) stopped = "errors";
    } else {
      consecutiveErrors = 0;
      counts[status === "generated" ? "generated" : "fallback"] += 1;
      log(`[${attempted}/${pending.length}] ${job.kind} ${job.id}: ${status} (${attempts} request${attempts === 1 ? "" : "s"})`);
    }
  }

  counts.remaining = pending.length - attempted;
  run.finishedAt = now().toISOString();
  run.counts = counts;
  run.stopped = stopped;
  await writeAtomic(runFile, `${JSON.stringify(run, null, 2)}\n`);
  log(`Done: generated ${counts.generated}, fallback ${counts.fallback}, errors ${counts.errors}, already generated ${counts.reused}, skipped ${counts.skipped}, not attempted ${counts.remaining}.`);
  if (stopped) log(`Stopped after ${MAX_CONSECUTIVE_ERRORS} errors in a row; run the same command again to continue.`);
  return { dryRun: false, outDir, plan, counts, stopped };
}
