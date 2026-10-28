import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { DEFAULT_SETTINGS, type OutputSettings } from "@arxiv-daily/core";
import dailyPromptZh from "../../packages/core/src/prompts/daily-paper-summary.system.md";
import detailPromptZh from "../../packages/core/src/prompts/paper-detail.system.md";
import { loadCliConfig } from "../../apps/cli/src/config";
import { assertWritableTarget, prepareOutDir } from "./evaluate";
import { checkSummaries, MAIN_RESULT_LENGTH_RANGE, monthlyMainResult, type MonthlyMainResult, type SummaryChecksResult } from "./summary-checks";
import { checkNotes, type NoteChecksResult } from "./summary-note-checks";
import { loadSummaryCorpus, type SummaryCorpus } from "./summary-corpus";
import { renderSummaryReport } from "./summary-report";

const REPORT_FILES = { report: "report.md", findings: "findings.json" } as const;

export interface SummaryCheckOptions {
  /** Vault root; defaults to the CLI config's `vault_root`. */
  vaultRoot?: string;
  /** CLI config supplying the vault root and output layout; the default config path when no vault is given. */
  configPath?: string;
  /** Relative to the vault; defaults to the config's (or the built-in) daily directory. */
  dailyDir?: string;
  papersDir?: string;
  from?: string;
  to?: string;
  outDir: string;
  /** Replace an existing report.md/findings.json in `outDir`. */
  force?: boolean;
}

export interface SummaryCheckDeps { now?: () => Date }

export interface PromptHash { file: string; sha256: string }

export interface SummaryCheckReport {
  schemaVersion: 1;
  generatedAt: string;
  scope: { vaultRoot: string; dailyDir: string; papersDir: string; range: { from: string | null; to: string | null } };
  /** The prompts as bundled when the check ran; reports and notes do not record which prompt wrote them. */
  prompts: { dailySummary: PromptHash; paperDetail: PromptHash };
  limits: { mainResultRange: { min: number; max: number } };
  corpus: Pick<SummaryCorpus, "counts" | "legacyReportDates" | "fallbacks">;
  daily: SummaryChecksResult;
  notes: NoteChecksResult;
  monthlyMainResult: MonthlyMainResult[];
}

export interface SummaryCheckResult { outDir: string; files: string[]; report: SummaryCheckReport }

export function parseSummaryCheckArgs(argv: readonly string[]): Partial<SummaryCheckOptions> & { help?: boolean } {
  const parsed: Partial<SummaryCheckOptions> & { help?: boolean } = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]!;
    if (flag === "--help" || flag === "-h") return { help: true };
    if (flag === "--force") { parsed.force = true; continue; }
    const take = (): string => {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) throw new Error(`${flag} needs a value`);
      index += 1;
      return value;
    };
    switch (flag) {
      case "--vault": parsed.vaultRoot = take(); break;
      case "--config": parsed.configPath = take(); break;
      case "--daily-dir": parsed.dailyDir = take(); break;
      case "--papers-dir": parsed.papersDir = take(); break;
      case "--out": parsed.outDir = take(); break;
      case "--from": parsed.from = requireDate(flag, take()); break;
      case "--to": parsed.to = requireDate(flag, take()); break;
      default: throw new Error(`unknown option: ${flag}`);
    }
  }
  if (!parsed.outDir) throw new Error("--out <directory> is required");
  if (parsed.from && parsed.to && parsed.from > parsed.to) {
    throw new Error(`--from ${parsed.from} is after --to ${parsed.to}; the start date is after the end date`);
  }
  return parsed;
}

function requireDate(flag: string, value: string): string {
  const date = /^\d{4}-\d{2}-\d{2}$/.test(value) ? new Date(`${value}T00:00:00Z`) : null;
  if (!date || Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== value) {
    throw new Error(`${flag} must be a date like 2026-10-14`);
  }
  return value;
}

const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");

/** Read the vault's daily entries and detail notes, run the offline checks and write report.md and findings.json. No model, no network. */
export async function runSummaryCheck(options: SummaryCheckOptions, deps: SummaryCheckDeps = {}): Promise<SummaryCheckResult> {
  const outDir = path.resolve(options.outDir);
  const files = [path.join(outDir, REPORT_FILES.report), path.join(outDir, REPORT_FILES.findings)];
  if (!options.force) {
    const existing = files.find((file) => existsSync(file));
    if (existing) throw new Error(`${existing} already exists; choose another --out or pass --force to replace it`);
  }
  await assertWritableTarget(outDir);

  const config = options.configPath || !options.vaultRoot
    ? await loadCliConfig({ configPath: options.configPath })
    : undefined;
  const vaultRoot = path.resolve(options.vaultRoot ?? config!.vaultRoot);
  const output: OutputSettings = config?.settings.output ?? DEFAULT_SETTINGS.output;
  const dailyDir = options.dailyDir ?? output.dailyDir;
  const papersDir = options.papersDir ?? output.papersDir;

  const corpus = await loadSummaryCorpus({ vaultRoot, dailyDir, papersDir, from: options.from, to: options.to });
  if (corpus.counts.reports === 0 && corpus.counts.notes === 0 && corpus.counts.legacyNotes === 0) {
    throw new Error(`no daily reports or detail notes found under ${vaultRoot} (looked in ${dailyDir} and ${papersDir}) for ${options.from ?? "the first day"} to ${options.to ?? "the last day"}`);
  }

  const report: SummaryCheckReport = {
    schemaVersion: 1,
    generatedAt: (deps.now?.() ?? new Date()).toISOString(),
    scope: { vaultRoot, dailyDir, papersDir, range: { from: options.from ?? null, to: options.to ?? null } },
    prompts: {
      dailySummary: { file: "packages/core/src/prompts/daily-paper-summary.system.md", sha256: sha256(dailyPromptZh) },
      paperDetail: { file: "packages/core/src/prompts/paper-detail.system.md", sha256: sha256(detailPromptZh) },
    },
    limits: { mainResultRange: { ...MAIN_RESULT_LENGTH_RANGE } },
    corpus: { counts: corpus.counts, legacyReportDates: corpus.legacyReportDates, fallbacks: corpus.fallbacks },
    daily: checkSummaries(corpus),
    notes: checkNotes(corpus.notes),
    monthlyMainResult: monthlyMainResult(corpus.entries),
  };

  await prepareOutDir(outDir);
  await writeFile(files[0]!, renderSummaryReport(report));
  await writeFile(files[1]!, `${JSON.stringify(report, null, 2)}\n`);
  return { outDir, files, report };
}
