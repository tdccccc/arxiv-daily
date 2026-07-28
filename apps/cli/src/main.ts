import { CliConfigError, loadCliConfig, type CliRuntimeConfig } from "./config";
import type { OperationRegistry, PipelineResult } from "@arxiv-daily/core";
import type { ManualFetchResult } from "@arxiv-daily/core";
import {
  deliverDailyEmailIfEnabled,
  resolveResendApiKey,
  validateFilterConfig,
  validateLlmConfig,
  formatDate,
  todayInTz,
  redactText,
} from "@arxiv-daily/core";
import type { HostAdapters } from "@arxiv-daily/core";
import type { CliIo, WritableTextStream } from "./main-types";
import { runInit } from "./init";
import { emailStatus, emailTest, emailVerifyStart } from "./email-cmd";
import {
  scheduleInstall,
  scheduleShow,
  scheduleUninstall,
} from "./schedule-cmd";
import { dataExport, dataImport } from "./data-cmd";
import { runUpdate } from "./update-cmd";
import { getCliVersion } from "./version";

export type { CliIo, WritableTextStream } from "./main-types";

type CliRunResult = PipelineResult | { kind: "skipped"; reason: string };

export interface CliCommandRuntime {
  pipeline: {
    runForDate(date: string): Promise<PipelineResult>;
  };
  scheduler?: {
    runForDateNow(date: string): Promise<CliRunResult>;
  };
  manualFetch: {
    fetchAndSummarize(id: string, date: string, signal?: AbortSignal): Promise<ManualFetchResult>;
  };
  operations?: OperationRegistry;
  host?: HostAdapters;
  settings?: CliRuntimeConfig["settings"];
}

export interface RunCliOptions {
  argv?: string[];
  env?: Record<string, string | undefined>;
  io?: CliIo;
  now?: () => Date;
  loadConfig?: typeof loadCliConfig;
  buildRuntime?: (
    config: CliRuntimeConfig,
  ) => CliCommandRuntime | Promise<CliCommandRuntime>;
  /** Test hooks for init / schedule / data */
  init?: typeof runInit;
  schedule?: {
    show?: typeof scheduleShow;
    install?: typeof scheduleInstall;
    uninstall?: typeof scheduleUninstall;
  };
  data?: {
    export?: typeof dataExport;
    import?: typeof dataImport;
  };
  update?: typeof runUpdate;
  isTTY?: boolean;
}

type CliCommand =
  | { name: "help" }
  | { name: "init" }
  | { name: "update"; checkOnly?: boolean; yes?: boolean }
  | { name: "run"; mode: "today" | "date" | "id"; date?: string; id?: string }
  | { name: "email"; sub: "test" | "status" | "verify-start"; date?: string }
  | { name: "schedule"; sub: "show" | "install" | "uninstall" }
  | { name: "data"; sub: "export"; out?: string }
  | { name: "data"; sub: "import"; zip?: string; yes?: boolean };

const USAGE = `Usage:
  arxiv-daily init
  arxiv-daily update [--check] [--yes]
  arxiv-daily run --today
  arxiv-daily run --date YYYY-MM-DD
  arxiv-daily run --id ARXIV_ID [--date YYYY-MM-DD]
  arxiv-daily email test [--date YYYY-MM-DD]
  arxiv-daily email status
  arxiv-daily email verify-start
  arxiv-daily schedule show
  arxiv-daily schedule install
  arxiv-daily schedule uninstall
  arxiv-daily data export --out PATH.zip
  arxiv-daily data import PATH.zip [--yes]
  arxiv-daily help

Config: $XDG_CONFIG_HOME/arxiv-daily/config.toml (run init first)
`;

export async function runCli(opts: RunCliOptions = {}): Promise<number> {
  const argv = opts.argv ?? process.argv.slice(2);
  const rawIo = opts.io ?? { stdout: process.stdout, stderr: process.stderr };
  let secrets: string[] = [];
  const io: CliIo = {
    stdout: { write: (chunk) => rawIo.stdout.write(redactText(String(chunk), { secrets })) },
    stderr: { write: (chunk) => rawIo.stderr.write(redactText(String(chunk), { secrets })) },
  };
  const env = { ...(opts.env ?? process.env) };
  const loadConfig = opts.loadConfig ?? loadCliConfig;
  const buildRuntime = opts.buildRuntime ?? defaultBuildRuntime;
  const now = opts.now ?? (() => new Date());

  let parsed: CliCommand;
  try {
    parsed = parseCli(argv);
  } catch (e) {
    writeLine(io.stderr, (e as Error).message);
    writeLine(io.stderr, USAGE.trimEnd());
    return 2;
  }

  if (parsed.name === "help") {
    writeLine(io.stdout, USAGE.trimEnd());
    writeLine(io.stdout, `Version: ${getCliVersion()}`);
    return 0;
  }

  if (parsed.name === "init") {
    const initFn = opts.init ?? runInit;
    return initFn({ env, stdout: io.stdout, stderr: io.stderr, isTTY: opts.isTTY });
  }

  if (parsed.name === "update") {
    const updateFn = opts.update ?? runUpdate;
    return updateFn(io, {
      checkOnly: parsed.checkOnly,
      yes: parsed.yes,
      isTTY: opts.isTTY ?? Boolean(process.stdin.isTTY),
    });
  }

  try {
    const config = await loadConfig({ env });
    secrets = [
      config.settings.llm.apiKey,
      config.settings.email.apiKey,
      config.settings.email.hostedToken,
    ].filter((value): value is string => Boolean(value));

    if (parsed.name === "schedule") {
      if (parsed.sub === "show") {
        return (opts.schedule?.show ?? scheduleShow)(config, io);
      }
      if (parsed.sub === "install") {
        return (opts.schedule?.install ?? scheduleInstall)(config, io);
      }
      return (opts.schedule?.uninstall ?? scheduleUninstall)(config, io);
    }

    if (parsed.name === "data") {
      if (parsed.sub === "export") {
        if (!parsed.out) {
          writeLine(io.stderr, "data export requires --out PATH.zip");
          return 2;
        }
        return (opts.data?.export ?? dataExport)(config, io, parsed.out);
      }
      if (!parsed.zip) {
        writeLine(io.stderr, "data import requires PATH.zip");
        return 2;
      }
      return (opts.data?.import ?? dataImport)(config, io, parsed.zip, {
        yes: parsed.yes,
        isTTY: opts.isTTY ?? Boolean(process.stdin.isTTY),
      });
    }

    if (parsed.name === "email" && parsed.sub === "status") {
      return emailStatus(config, io);
    }

    const validation =
      parsed.name === "run" && parsed.mode === "id"
        ? validateLlmConfig(config.settings)
        : parsed.name === "email"
          ? { ok: true as const, reasons: [] as string[] }
          : validateFilterConfig(config.settings);
    if (!validation.ok) {
      writeLine(io.stderr, `Invalid config:\n${validation.reasons.join("\n")}`);
      return 2;
    }

    const runtime = await buildRuntime(config);
    const removeSignalHandlers = installSignalHandlers(runtime.operations, io);
    try {
      if (parsed.name === "email") {
        if (!runtime.host) {
          writeLine(io.stderr, "email commands require host adapters");
          return 1;
        }
        if (parsed.sub === "test") {
          return emailTest(config, runtime.host, io, parsed.date, now);
        }
        return emailVerifyStart(config, runtime.host, io);
      }

      if (parsed.name === "run") {
        if (parsed.mode === "id") {
          if (!parsed.id) throw new Error("run --id requires an arXiv id");
          const date =
            parsed.date ??
            formatDate(todayInTz(now(), config.settings.arxiv.timezone));
          const operation = runtime.operations?.begin(
            "detail-summary",
            `Detail summary: ${parsed.id}`,
            parsed.id,
          );
          try {
            const result = operation
              ? await runtime.manualFetch.fetchAndSummarize(
                  parsed.id,
                  date,
                  operation.signal,
                )
              : await runtime.manualFetch.fetchAndSummarize(parsed.id, date);
            return writeManualFetchResult(io, result);
          } finally {
            operation?.finish();
          }
        }

        const date =
          parsed.mode === "today"
            ? formatDate(todayInTz(now(), config.settings.arxiv.timezone))
            : parsed.date;
        if (!date) throw new Error("run requires --today or --date");

        const result = runtime.scheduler
          ? await runtime.scheduler.runForDateNow(date)
          : await runtime.pipeline.runForDate(date);
        if (
          !runtime.scheduler &&
          result.kind === "completed" &&
          result.digest &&
          runtime.host
        ) {
          await deliverDailyEmailIfEnabled(result.digest, {
            storage: runtime.host.storage,
            http: runtime.host.http,
            output: config.settings.output,
            email: config.settings.email,
            apiKey: resolveResendApiKey(config.settings.email, {}),
          });
        }
        return writeRunResult(io, date, result);
      }
    } finally {
      removeSignalHandlers();
    }
  } catch (e) {
    writeLine(io.stderr, (e as Error).message);
    return e instanceof CliConfigError ? 2 : 1;
  }

  writeLine(io.stderr, "internal: unhandled command");
  return 1;
}

function parseCli(argv: string[]): CliCommand {
  const rest: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === undefined) continue;
    if (arg === "--help" || arg === "-h") return { name: "help" };
    if (arg === "--config" || arg === "--vault-root" || arg === "--cache-dir") {
      throw new Error(
        `${arg} is no longer supported; use ~/.config/arxiv-daily/config.toml (arxiv-daily init)`,
      );
    }
    rest.push(arg);
  }

  const [commandName, ...commandArgs] = rest;
  if (!commandName || commandName === "help") return { name: "help" };
  if (commandName === "init") return { name: "init" };

  if (commandName === "update") {
    return {
      name: "update",
      checkOnly: commandArgs.includes("--check"),
      yes: commandArgs.includes("--yes") || commandArgs.includes("-y"),
    };
  }

  if (commandName === "run") {
    const today = commandArgs.includes("--today");
    const date = optionValue(commandArgs, "--date");
    const id = optionValue(commandArgs, "--id");
    // --id may also take --date for note dating
    if (id) {
      if (today) throw new Error("run --id cannot be combined with --today");
      return { name: "run", mode: "id", id, date };
    }
    if (today && date) throw new Error("run --today cannot be combined with --date");
    if (today) return { name: "run", mode: "today" };
    if (date) return { name: "run", mode: "date", date };
    throw new Error("run requires --today, --date YYYY-MM-DD, or --id ARXIV_ID");
  }

  if (commandName === "email" || commandName === "email-test") {
    if (commandName === "email-test") {
      return {
        name: "email",
        sub: "test",
        date: optionValue(commandArgs, "--date"),
      };
    }
    const sub = commandArgs[0];
    if (sub === "test") {
      return {
        name: "email",
        sub: "test",
        date: optionValue(commandArgs.slice(1), "--date"),
      };
    }
    if (sub === "status") return { name: "email", sub: "status" };
    if (sub === "verify-start") return { name: "email", sub: "verify-start" };
    throw new Error('email requires subcommand: test | status | verify-start');
  }

  if (commandName === "schedule") {
    const sub = commandArgs[0];
    if (sub === "show" || sub === "install" || sub === "uninstall") {
      return { name: "schedule", sub };
    }
    throw new Error("schedule requires subcommand: show | install | uninstall");
  }

  if (commandName === "data") {
    const sub = commandArgs[0];
    if (sub === "export") {
      return {
        name: "data",
        sub: "export",
        out: optionValue(commandArgs.slice(1), "--out"),
      };
    }
    if (sub === "import") {
      const args = commandArgs.slice(1);
      const yes = args.includes("--yes");
      const zip = args.find((a) => !a.startsWith("--"));
      return { name: "data", sub: "import", zip, yes };
    }
    throw new Error("data requires subcommand: export | import");
  }

  if (commandName === "run-pending") {
    throw new Error(
      "run-pending was removed; use: arxiv-daily run --today (or run --date YYYY-MM-DD)",
    );
  }
  if (commandName === "summarize") {
    throw new Error("summarize was removed; use: arxiv-daily run --id ARXIV_ID");
  }

  throw new Error(`Unknown command: ${commandName}`);
}

function optionValue(argv: string[], option: string): string | undefined {
  const index = argv.indexOf(option);
  if (index < 0) return undefined;
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`${option} requires a value`);
  }
  return value;
}

function writeRunResult(io: CliIo, date: string, result: CliRunResult): number {
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
    writeLine(io.stdout, `run --id: wrote ${result.path}`);
    return 0;
  }
  if (result.kind === "already_exists") {
    writeLine(io.stdout, `run --id: already exists ${result.path}`);
    return 0;
  }
  writeLine(io.stderr, `run --id: ${result.kind} (${result.reason})`);
  return 1;
}

function writeLine(stream: WritableTextStream, line: string): void {
  stream.write(`${line}\n`);
}

function installSignalHandlers(
  operations: OperationRegistry | undefined,
  io: CliIo,
): () => void {
  if (!operations || typeof process === "undefined" || !process.on) return () => {};
  let signalCount = 0;
  const handler = (signal: NodeJS.Signals) => {
    signalCount += 1;
    if (signalCount === 1) {
      const active = operations.snapshot();
      operations.cancelAll(`cancelled by ${signal}`);
      writeLine(
        io.stderr,
        `arxiv-daily: ${signal} received; cancelling ${active.length} active task${active.length === 1 ? "" : "s"} and waiting`,
      );
      return;
    }
    process.exit(128 + (signal === "SIGINT" ? 2 : 15));
  };
  process.on("SIGINT", handler);
  process.on("SIGTERM", handler);
  return () => {
    process.off("SIGINT", handler);
    process.off("SIGTERM", handler);
  };
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
