import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { LlmClient, Logger, type LlmCallMetrics, type LlmSettings } from "@arxiv-daily/core";
import { NodeHttpClient } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../../apps/cli/src/config";
import { readModelSettings } from "../acceptance/exploration-model";
import { prepareOutDir } from "./evaluate";
import { redactor } from "./rerun";
import { buildJudgeCorrection, checkJudgeOutput, type Judgment } from "./summary-judge-contract";
import { prepareRegeneratedJudgeInput } from "./summary-judge-regenerated";
import { prepareJudgeInput, type JudgeItem, type PrepareJudgeOptions, type SkippedItem } from "./summary-judge-input";
import { requireDate } from "./summary-check";
import { writeJudgeReport } from "./summary-judge-report";

/** Characters per token used for the size estimate: a mix of English paper text and Chinese summaries. */
export const CHARS_PER_TOKEN = 3;
const MAX_CONSECUTIVE_ERRORS = 3;
const REQUESTS_PER_ITEM = 2;
const SHOWN_SKIPPED = 10;

export interface SummaryJudgeOptions extends Omit<PrepareJudgeOptions, "topics"> {
  /** `[llm]`-only TOML for the judge model (the format of the acceptance `--model-config`). */
  judgeConfigPath: string;
  /** The generator's model name when it cannot be read from a CLI config. */
  generatorModel?: string;
  outDir: string;
  yes?: boolean;
  /** Judge again even when a judgment for the same judge, prompt and source exists. */
  rejudge?: boolean;
  /** `findings.json` of summary-check run with `--sources`: the report then compares its unfound numbers with the judgments. */
  offlinePath?: string;
  /** The `--out` directory of summary-regenerate.mjs: judge those regenerated summaries instead of the vault's. */
  summariesDir?: string;
  /** Only rebuild report.md and judgments.json from what is in `outDir`. */
  reportOnly?: boolean;
}

export interface JudgeModelLike {
  call(
    messages: Array<{ role: "system" | "user" | "assistant"; content: string }>,
    options?: { temperature?: number; onMetrics?: (metrics: LlmCallMetrics) => void },
  ): Promise<string>;
}

export interface GeneratorConfig { model: string; topics: Array<{ tag: string; name: string }> }

export interface SummaryJudgeDeps {
  /** The judge model client; tests pass a fake, otherwise one is built from the judge config. */
  llm?: JudgeModelLike;
  now?: () => Date;
  log?: (line: string) => void;
  /** Reads the generator's model and topics from a CLI config (the default one when no path is given); null when there is none. */
  loadGeneratorConfig?: (configPath?: string) => Promise<GeneratorConfig | null>;
}

export interface JudgePlan {
  items: { daily: number; note: number };
  skipped: SkippedItem[];
  alreadyJudged: number;
  toJudge: number;
  maxRequests: number;
  totalChars: number;
  estimatedTokens: number;
}

export interface JudgeCounts { judged: number; invalid: number; errors: number; reused: number; skipped: number; remaining: number }

export interface SummaryJudgeResult {
  dryRun: boolean;
  outDir: string;
  plan: JudgePlan;
  counts: JudgeCounts | null;
  stopped: "errors" | null;
}

export function parseSummaryJudgeArgs(argv: readonly string[]): Partial<SummaryJudgeOptions> & { help?: boolean } {
  const parsed: Partial<SummaryJudgeOptions> & { help?: boolean } = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]!;
    if (flag === "--help" || flag === "-h") return { help: true };
    if (flag === "--yes") { parsed.yes = true; continue; }
    if (flag === "--rejudge") { parsed.rejudge = true; continue; }
    if (flag === "--report-only") { parsed.reportOnly = true; continue; }
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
      case "--judge-config": parsed.judgeConfigPath = take(); break;
      case "--config": parsed.configPath = take(); break;
      case "--generator-model": parsed.generatorModel = take(); break;
      case "--vault": parsed.vaultRoot = take(); break;
      case "--daily-dir": parsed.dailyDir = take(); break;
      case "--papers-dir": parsed.papersDir = take(); break;
      case "--sources": parsed.sourcesDir = take(); break;
      case "--out": parsed.outDir = take(); break;
      case "--offline": parsed.offlinePath = take(); break;
      case "--summaries": parsed.summariesDir = take(); break;
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
  if (!parsed.outDir) throw new Error("--out <directory> is required");
  if (!parsed.reportOnly) {
    if (!parsed.judgeConfigPath) throw new Error("--judge-config <file> is required");
    if (!parsed.sourcesDir) throw new Error("--sources <directory> is required");
  }
  if (parsed.from && parsed.to && parsed.from > parsed.to) {
    throw new Error(`--from ${parsed.from} is after --to ${parsed.to}; the start date is after the end date`);
  }
  return parsed;
}

async function defaultLoadGeneratorConfig(configPath?: string): Promise<GeneratorConfig | null> {
  try {
    const config = await loadCliConfig({ configPath });
    return { model: config.settings.llm.model, topics: config.settings.arxiv.topics.map((topic) => ({ tag: topic.tag, name: topic.name })) };
  } catch (error) {
    // An explicit path that fails is the user's to fix; a missing default config just means "unknown".
    if (configPath) throw error;
    return null;
  }
}

const sameModel = (left: string, right: string): boolean => left.trim().toLowerCase() === right.trim().toLowerCase();
const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");
const fileFor = (outDir: string, item: Pick<JudgeItem, "kind" | "id">): string => path.join(outDir, "judgments", `${item.kind}-${item.id}.json`);

async function writeAtomic(file: string, text: string): Promise<void> {
  const temporary = `${file}.tmp`;
  await writeFile(temporary, text);
  await rename(temporary, file);
}

interface JudgmentRecord {
  status: "judged" | "invalid" | "error";
  judge: { provider: string; model: string };
  judgePromptHash: string;
  summaryHash?: string;
  item: { source: { contentHash: string } };
}

/** A finished judgment stands when the judge, its prompt and the source text are all the same. */
async function isCurrent(file: string, item: JudgeItem, judge: LlmSettings, promptHash: string): Promise<boolean> {
  try {
    const record = JSON.parse(await readFile(file, "utf8")) as JudgmentRecord;
    return (record.status === "judged" || record.status === "invalid")
      && record.judge.model === judge.model && record.judge.provider === judge.provider
      && record.judgePromptHash === promptHash && record.item.source.contentHash === item.source.contentHash
      && record.summaryHash === item.summaryHash;
  } catch {
    return false;
  }
}

function usageOf(metrics: LlmCallMetrics | undefined): { inputTokens?: number; outputTokens?: number; totalTokens?: number; usageComplete: boolean } | null {
  if (!metrics) return null;
  return { inputTokens: metrics.inputTokens, outputTokens: metrics.outputTokens, totalTokens: metrics.totalTokens, usageComplete: metrics.usageComplete };
}

interface Attempt { raw: string; reasons: string[] | null; usage: ReturnType<typeof usageOf>; elapsedMs: number }

/** Judge one item: one call, and when the reply fails validation one retry that shows the model its reasons. */
async function judgeItem(
  item: JudgeItem,
  llm: JudgeModelLike,
  now: () => Date,
): Promise<{ attempts: Attempt[]; judgment: Judgment | null; error?: unknown }> {
  const attempts: Attempt[] = [];
  let messages = item.messages;
  for (let attempt = 1; attempt <= REQUESTS_PER_ITEM; attempt += 1) {
    const started = now().getTime();
    let metrics: LlmCallMetrics | undefined;
    let raw: string;
    try {
      raw = await llm.call(messages, { temperature: 0, onMetrics: (observed) => { metrics = observed; } });
    } catch (error) {
      return { attempts, judgment: null, error };
    }
    const checked = checkJudgeOutput(raw, item.context);
    attempts.push({ raw, reasons: checked.ok ? null : checked.reasons, usage: usageOf(metrics), elapsedMs: now().getTime() - started });
    if (checked.ok) return { attempts, judgment: checked.judgment };
    messages = [...messages, { role: "assistant", content: raw }, { role: "user", content: buildJudgeCorrection(checked.reasons) }];
  }
  return { attempts, judgment: null };
}

/**
 * Judge a sample of daily entries and detail notes against their cached sources with a separately configured model.
 * Without `yes` it only prints the plan. Each finished item is written to `<out>/judgments/` as it completes,
 * so an interrupted run resumes where it stopped.
 */
export async function runSummaryJudge(options: SummaryJudgeOptions, deps: SummaryJudgeDeps = {}): Promise<SummaryJudgeResult> {
  const log = deps.log ?? (() => undefined);
  const now = deps.now ?? (() => new Date());
  const outDir = path.resolve(options.outDir);

  const judge = await readModelSettings(options.judgeConfigPath, { label: "Judge model config", hint: "Pass --judge-config <file> with an [llm] table (api_key, base_url, model)." });
  if (options.summariesDir && (options.ids || options.daily !== undefined || options.notes !== undefined || options.includeOlderNotes || options.from || options.to)) {
    throw new Error("--summaries judges the sample recorded by the regenerate run, so --daily, --notes, --ids, --include-older-notes, --from and --to cannot be combined with it.");
  }
  // With --summaries the regenerate run names its own generator; the CLI config is read only when --config is given.
  const regenerated = options.summariesDir ? await prepareRegeneratedJudgeInput({ summariesDir: options.summariesDir, sourcesDir: options.sourcesDir }) : null;
  const generator = regenerated && !options.configPath ? null : await (deps.loadGeneratorConfig ?? defaultLoadGeneratorConfig)(options.configPath);
  const generatorModel = options.generatorModel ?? regenerated?.generator.model ?? generator?.model;
  if (!generatorModel?.trim()) {
    throw new Error("The generator model could not be determined, so it cannot be compared with the judge model. Pass --config <CLI config.toml> or --generator-model <name>.");
  }
  if (sameModel(judge.model, generatorModel)) {
    throw new Error(`The judge must be a different model from the one that wrote the summaries, but both are "${judge.model.trim()}". Use another model in --judge-config.`);
  }

  const prepared = regenerated ?? await prepareJudgeInput({ ...options, topics: generator?.topics });
  const hashes = new Map(prepared.items.map((item) => [item, sha256(item.messages[0]!.content)]));
  const pending: JudgeItem[] = [];
  let reused = 0;
  for (const item of prepared.items) {
    if (!options.rejudge && await isCurrent(fileFor(outDir, item), item, judge, hashes.get(item)!)) reused += 1;
    else pending.push(item);
  }
  const totalChars = pending.reduce((sum, item) => sum + item.messages.reduce((size, message) => size + message.content.length, 0), 0);
  const plan: JudgePlan = {
    items: { daily: prepared.items.filter((item) => item.kind === "daily").length, note: prepared.items.filter((item) => item.kind === "note").length },
    skipped: prepared.skipped, alreadyJudged: reused, toJudge: pending.length,
    maxRequests: pending.length * REQUESTS_PER_ITEM, totalChars, estimatedTokens: Math.ceil(totalChars / CHARS_PER_TOKEN),
  };

  log(`Judge model: ${judge.model} (${judge.provider}); generator model: ${generatorModel.trim()}.`);
  log(`Items: daily ${plan.items.daily}, note ${plan.items.note} (${prepared.items.length} items). Already judged with this judge, prompt and source: ${reused}. To judge: ${pending.length}.`);
  if (prepared.skipped.length > 0) {
    log(`Skipped ${prepared.skipped.length} (not replaced): ${prepared.skipped.slice(0, SHOWN_SKIPPED).map((entry) => `${entry.id} [${entry.reason}]`).join(", ")}${prepared.skipped.length > SHOWN_SKIPPED ? ", ..." : ""}`);
  }
  log(`This ${options.yes ? "will" : "would"} make at most ${plan.maxRequests} requests (${pending.length} items x ${REQUESTS_PER_ITEM}: one retry each when the reply fails validation), sending ${totalChars} characters in the first requests, about ${plan.estimatedTokens} input tokens (estimated at ${CHARS_PER_TOKEN} characters per token for mixed English and Chinese text).`);
  if (!options.yes) {
    log("No model was called and nothing was written. Add --yes to judge.");
    return { dryRun: true, outDir, plan, counts: null, stopped: null };
  }

  await prepareOutDir(outDir);
  await mkdir(path.join(outDir, "judgments"), { recursive: true });
  const redact = redactor(judge);
  const llm = deps.llm ?? (() => {
    const logger = new Logger("error");
    logger.setSensitiveValues([judge.apiKey]);
    return new LlmClient(judge, logger, new NodeHttpClient());
  })();

  const startedAt = now().toISOString();
  const counts: JudgeCounts = { judged: 0, invalid: 0, errors: 0, reused, skipped: prepared.skipped.length, remaining: 0 };
  const runFile = path.join(outDir, "run.json");
  const run = {
    schemaVersion: 1, startedAt, finishedAt: null as string | null,
    // The judge config path and the secrets it holds are not recorded.
    args: { daily: options.daily ?? null, notes: options.notes ?? null, ids: options.ids ?? null, from: options.from ?? null, to: options.to ?? null, includeOlderNotes: Boolean(options.includeOlderNotes), rejudge: Boolean(options.rejudge), sourcesDir: path.resolve(options.sourcesDir) },
    seed: options.seed ?? "summary-judge",
    summaries: options.summariesDir ? { dir: path.resolve(options.summariesDir), generator: regenerated!.generator } : null,
    vault: options.summariesDir ? null : prepared.items[0]?.origin.kind === "historical" ? prepared.items[0].origin.vault : null,
    requested: prepared.requested, allocation: prepared.allocation,
    judge: { provider: judge.provider, model: judge.model, thinkingMode: judge.thinkingMode, reasoningEffort: judge.reasoningEffort },
    generatorModel: generatorModel.trim(),
    sample: { daily: prepared.sample.daily, notes: prepared.sample.notes },
    skipped: prepared.skipped, plan: { ...plan, skipped: undefined }, counts: null as JudgeCounts | null, stopped: null as "errors" | null,
  };
  await writeAtomic(runFile, `${JSON.stringify(run, null, 2)}\n`);

  let consecutiveErrors = 0;
  let stopped: "errors" | null = null;
  let attempted = 0;
  for (const item of pending) {
    if (stopped) break;
    attempted += 1;
    const itemStart = now();
    const outcome = await judgeItem(item, llm, now);
    const base = {
      schemaVersion: 1,
      item: { kind: item.kind, id: item.id, title: item.title, date: item.date, topic: item.topic, ...(item.version ? { version: item.version } : {}), source: item.source },
      judge: { provider: judge.provider, model: judge.model, thinkingMode: judge.thinkingMode, reasoningEffort: judge.reasoningEffort },
      judgePromptHash: hashes.get(item)!,
      summaryHash: item.summaryHash, origin: item.origin, summary: { locations: item.locations },
      attempts: outcome.attempts, startedAt: itemStart.toISOString(),
    };
    const finish = { finishedAt: now().toISOString(), elapsedMs: now().getTime() - itemStart.getTime() };
    let status: "judged" | "invalid" | "error";
    let record: Record<string, unknown>;
    if (outcome.error !== undefined) {
      status = "error";
      const message = redact((outcome.error as Error)?.message ?? outcome.error) || "unknown error";
      record = { ...base, status, invalid: false, judgment: null, error: message, ...finish };
      counts.errors += 1;
      consecutiveErrors += 1;
      log(`[${attempted}/${pending.length}] ${item.kind} ${item.id}: error: ${message}`);
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) stopped = "errors";
    } else if (outcome.judgment) {
      status = "judged";
      record = { ...base, status, invalid: false, judgment: outcome.judgment, ...finish };
      counts.judged += 1;
      consecutiveErrors = 0;
      log(`[${attempted}/${pending.length}] ${item.kind} ${item.id}: judged (${outcome.attempts.length} attempt${outcome.attempts.length === 1 ? "" : "s"})`);
    } else {
      status = "invalid";
      record = { ...base, status, invalid: true, judgment: null, ...finish };
      counts.invalid += 1;
      consecutiveErrors = 0;
      log(`[${attempted}/${pending.length}] ${item.kind} ${item.id}: invalid after ${outcome.attempts.length} attempts`);
    }
    await writeAtomic(fileFor(outDir, item), `${JSON.stringify(record, null, 2)}\n`);
  }

  counts.remaining = pending.length - attempted;
  run.finishedAt = now().toISOString();
  run.counts = counts;
  run.stopped = stopped;
  await writeAtomic(runFile, `${JSON.stringify(run, null, 2)}\n`);
  const report = await writeJudgeReport(outDir, { now, offlinePath: options.offlinePath });
  log(`Report: ${report.files.join(" and ")}`);
  log(`Done: judged ${counts.judged}, invalid ${counts.invalid}, errors ${counts.errors}, already judged ${counts.reused}, skipped ${counts.skipped}, not attempted ${counts.remaining}.`);
  if (stopped) log(`Stopped after ${MAX_CONSECUTIVE_ERRORS} errors in a row; run the same command again to continue.`);
  return { dryRun: false, outDir, plan, counts, stopped };
}
