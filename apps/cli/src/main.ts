import { CliConfigError, loadCliConfig, type CliRuntimeConfig } from "./config";
import type { PipelineResult } from "@arxiv-daily/core";
import type { ManualFetchResult } from "@arxiv-daily/core";
import {
  validateFilterConfig,
  validateLlmConfig,
} from "@arxiv-daily/core";
import { daysBefore, formatDate, todayInTz } from "@arxiv-daily/core";

type CliRunResult = PipelineResult | { kind: "skipped"; reason: string };

export interface WritableTextStream {
  write(chunk: string): unknown;
}

export interface CliIo {
  stdout: WritableTextStream;
  stderr: WritableTextStream;
}

export interface CliCommandRuntime {
  pipeline: {
    runForDate(date: string): Promise<PipelineResult>;
  };
  scheduler?: {
    runForDateNow(date: string): Promise<CliRunResult>;
    runAllPending(): Promise<Array<{ date: string; result: CliRunResult }>>;
  };
  manualFetch: {
    fetchAndSummarize(id: string, date: string): Promise<ManualFetchResult>;
  };
}

export interface RunCliOptions {
  argv?: string[];
  cwd?: string;
  env?: Record<string, string | undefined>;
  io?: CliIo;
  now?: () => Date;
  loadConfig?: typeof loadCliConfig;
  buildRuntime?: (
    config: CliRuntimeConfig,
  ) => CliCommandRuntime | Promise<CliCommandRuntime>;
}

type CliCommand =
  | { name: "help" }
  | { name: "run"; date?: string }
  | { name: "run-pending" }
  | { name: "summarize"; id?: string; date?: string };

interface ParsedCli {
  command: CliCommand;
  configPath?: string;
  vaultRoot?: string;
  cacheDir?: string;
}

const USAGE = `Usage:
  arxiv-daily run --date YYYY-MM-DD [--config path] [--vault-root path]
  arxiv-daily run-pending [--config path] [--vault-root path]
  arxiv-daily summarize --id ARXIV_ID [--date YYYY-MM-DD]

Options:
  --config path       JSON config file (default: arxiv-daily.config.json)
  --vault-root path   Workspace/vault root for generated files
  --cache-dir path    HTML cache directory
  --help              Show this help
`;

export async function runCli(opts: RunCliOptions = {}): Promise<number> {
  const argv = opts.argv ?? process.argv.slice(2);
  const io = opts.io ?? { stdout: process.stdout, stderr: process.stderr };
  const env = { ...(opts.env ?? process.env) };
  const loadConfig = opts.loadConfig ?? loadCliConfig;
  const buildRuntime = opts.buildRuntime ?? defaultBuildRuntime;
  const now = opts.now ?? (() => new Date());

  let parsed: ParsedCli;
  try {
    parsed = parseCli(argv);
  } catch (e) {
    writeLine(io.stderr, (e as Error).message);
    writeLine(io.stderr, USAGE.trimEnd());
    return 2;
  }

  if (parsed.command.name === "help") {
    writeLine(io.stdout, USAGE.trimEnd());
    return 0;
  }

  if (parsed.vaultRoot) env.ARXIV_DAILY_VAULT_ROOT = parsed.vaultRoot;
  if (parsed.cacheDir) env.ARXIV_DAILY_CACHE_DIR = parsed.cacheDir;

  try {
    const config = await loadConfig({
      cwd: opts.cwd,
      configPath: parsed.configPath,
      env,
    });
    const validation =
      parsed.command.name === "summarize"
        ? validateLlmConfig(config.settings)
        : validateFilterConfig(config.settings);
    if (!validation.ok) {
      writeLine(io.stderr, `Invalid config:\n${validation.reasons.join("\n")}`);
      return 2;
    }

    const runtime = await buildRuntime(config);
    if (parsed.command.name === "run") {
      if (!parsed.command.date) throw new Error("run requires --date");
      const result = runtime.scheduler
        ? await runtime.scheduler.runForDateNow(parsed.command.date)
        : await runtime.pipeline.runForDate(parsed.command.date);
      return writeRunResult(io, parsed.command.date, result);
    }
    if (parsed.command.name === "run-pending") {
      if (runtime.scheduler) {
        const results = await runtime.scheduler.runAllPending();
        let failed = false;
        for (const { date, result } of results) {
          const code = writeRunResult(io, date, result);
          if (code !== 0) failed = true;
        }
        return failed ? 1 : 0;
      }
      const dates = pendingDates(config, now());
      let failed = false;
      for (const date of dates) {
        const result = await runtime.pipeline.runForDate(date);
        const code = writeRunResult(io, date, result);
        if (code !== 0) failed = true;
      }
      return failed ? 1 : 0;
    }
    if (!parsed.command.id) throw new Error("summarize requires --id");
    const date =
      parsed.command.date ??
      formatDate(todayInTz(now(), config.settings.arxiv.timezone));
    const result = await runtime.manualFetch.fetchAndSummarize(
      parsed.command.id,
      date,
    );
    return writeManualFetchResult(io, result);
  } catch (e) {
    writeLine(io.stderr, (e as Error).message);
    return e instanceof CliConfigError ? 2 : 1;
  }
}

function parseCli(argv: string[]): ParsedCli {
  const global: Omit<ParsedCli, "command"> = {};
  const rest: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === undefined) continue;
    if (arg === "--help" || arg === "-h") return { command: { name: "help" } };
    if (arg === "--config") {
      global.configPath = requireValue(argv, ++i, arg);
      continue;
    }
    if (arg === "--vault-root") {
      global.vaultRoot = requireValue(argv, ++i, arg);
      continue;
    }
    if (arg === "--cache-dir") {
      global.cacheDir = requireValue(argv, ++i, arg);
      continue;
    }
    rest.push(arg);
  }

  const [commandName, ...commandArgs] = rest;
  if (!commandName || commandName === "help") {
    return { ...global, command: { name: "help" } };
  }
  if (commandName === "run") {
    return {
      ...global,
      command: { name: "run", date: optionValue(commandArgs, "--date") },
    };
  }
  if (commandName === "run-pending") {
    return { ...global, command: { name: "run-pending" } };
  }
  if (commandName === "summarize") {
    return {
      ...global,
      command: {
        name: "summarize",
        id: optionValue(commandArgs, "--id"),
        date: optionValue(commandArgs, "--date"),
      },
    };
  }
  throw new Error(`Unknown command: ${commandName}`);
}

function requireValue(argv: string[], index: number, option: string): string {
  const value = argv[index];
  if (!value || value.startsWith("--")) {
    throw new Error(`${option} requires a value`);
  }
  return value;
}

function optionValue(argv: string[], option: string): string | undefined {
  const index = argv.indexOf(option);
  if (index < 0) return undefined;
  return requireValue(argv, index + 1, option);
}

function pendingDates(config: CliRuntimeConfig, now: Date): string[] {
  const lookbackDays = 5; // LOOKBACK_DAYS constant
  const timezone = config.settings.arxiv.timezone;
  const today = todayInTz(now, timezone);
  const dates: string[] = [];
  for (let i = lookbackDays - 1; i >= 0; i--) {
    dates.push(formatDate(daysBefore(today, i, timezone)));
  }
  return dates;
}

function writeRunResult(
  io: CliIo,
  date: string,
  result: CliRunResult,
): number {
  if (result.kind === "completed") {
    writeLine(
      io.stdout,
      `run ${date}: completed (${result.papersWritten} papers written)`,
    );
    return 0;
  }
  if (result.kind === "skipped") {
    writeLine(io.stdout, `run ${date}: skipped (${result.reason})`);
    return 0;
  }
  writeLine(io.stderr, `run ${date}: ${result.kind} (${result.reason})`);
  return 1;
}

function writeManualFetchResult(io: CliIo, result: ManualFetchResult): number {
  if (result.kind === "done") {
    writeLine(io.stdout, `summarize: wrote ${result.path}`);
    return 0;
  }
  if (result.kind === "already_exists") {
    writeLine(io.stdout, `summarize: already exists ${result.path}`);
    return 0;
  }
  writeLine(io.stderr, `summarize: ${result.kind} (${result.reason})`);
  return 1;
}

function writeLine(stream: WritableTextStream, line: string): void {
  stream.write(`${line}\n`);
}

async function defaultBuildRuntime(
  config: CliRuntimeConfig,
): Promise<CliCommandRuntime> {
  const { buildCliRuntime } = await import("./runtime");
  return buildCliRuntime(config);
}

if (typeof require !== "undefined" && require.main === module) {
  void runCli()
    .then((code) => {
      process.exitCode = code;
    })
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
