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
import { inspectPapers, inspectProduct } from "./inspect-cmd";
import { runCliLibrary } from "./library-cmd";
import { runWorkbench } from "./workbench/launch";
import { validateFrameOrigin } from "./workbench/embedding";

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
  dispose?: () => void;
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
  ui?: typeof runWorkbench;
  isTTY?: boolean;
}

type CliCommand =
  | { name: "help" }
  | { name: "init" }
  | { name: "status" }
  | { name: "ui"; port: number; open: boolean; frameOrigin?: string }
  | { name: "papers"; query: string; offset: number; limit: number }
  | { name: "library"; args: string[] }
  | { name: "update"; checkOnly?: boolean; yes?: boolean }
  | { name: "run"; mode: "today" | "date" | "id"; date?: string; id?: string }
  | { name: "email"; sub: "test" | "status" | "verify-start"; date?: string }
  | { name: "schedule"; sub: "show" | "install" | "uninstall" }
  | { name: "data"; sub: "export"; out?: string }
  | { name: "data"; sub: "import"; zip?: string; yes?: boolean };

const USAGE = `Usage:
  arxiv-daily init
  arxiv-daily status
  arxiv-daily ui [--port PORT] [--no-open] [--frame-origin ORIGIN]
  arxiv-daily papers [--query TEXT] [--offset N] [--limit N]
  arxiv-daily library connect PATH
  arxiv-daily library status|prepare|scan|index|propose|directions|update|revoke
  arxiv-daily library authorize --fingerprint HASH
  arxiv-daily library confirm --candidate ID --proposal-revision N --profile-revision N
  arxiv-daily library search --query TEXT [--mode hybrid|lexical|dense] [--limit N]
  arxiv-daily library review --input REQUEST.json
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

  try {
    if (parsed.name === "init") {
      const initFn = opts.init ?? runInit;
      return await initFn({ env, stdout: io.stdout, stderr: io.stderr, isTTY: opts.isTTY });
    }

    if (parsed.name === "update") {
      const updateFn = opts.update ?? runUpdate;
      return await updateFn(io, {
        checkOnly: parsed.checkOnly,
        yes: parsed.yes,
        isTTY: opts.isTTY ?? Boolean(process.stdin.isTTY),
      });
    }

    let config: CliRuntimeConfig;
    try { config = await loadConfig({ env }); }
    catch (error) {
      if (parsed.name === "ui" && error instanceof CliConfigError && (error.cause as NodeJS.ErrnoException)?.code === "ENOENT") {
        return await (opts.ui ?? runWorkbench)(undefined, io, { port: parsed.port, open: parsed.open, env, ...(parsed.frameOrigin ? { frameOrigin: parsed.frameOrigin } : {}) });
      }
      throw error;
    }
    secrets = [
      config.settings.llm.apiKey,
      config.settings.email.apiKey,
      config.settings.email.hostedToken,
      config.settings.embedding.apiKey,
    ].filter((value): value is string => Boolean(value));

    if (parsed.name === "status") {
      writeLine(io.stdout, JSON.stringify(await inspectProduct(config)));
      return 0;
    }
    if (parsed.name === "ui") {
      return await (opts.ui ?? runWorkbench)(config, io, { port: parsed.port, open: parsed.open, env, ...(parsed.frameOrigin ? { frameOrigin: parsed.frameOrigin } : {}) });
    }
    if (parsed.name === "papers") {
      writeLine(io.stdout, JSON.stringify(await inspectPapers(config, parsed.query, parsed.offset, parsed.limit)));
      return 0;
    }
    if (parsed.name === "library") return await runCliLibrary(config, parsed.args, io);

    if (parsed.name === "schedule") {
      if (parsed.sub === "show") {
        return await (opts.schedule?.show ?? scheduleShow)(config, io);
      }
      if (parsed.sub === "install") {
        return await (opts.schedule?.install ?? scheduleInstall)(config, io);
      }
      return await (opts.schedule?.uninstall ?? scheduleUninstall)(config, io);
    }

    if (parsed.name === "data") {
      if (parsed.sub === "export") {
        if (!parsed.out) {
          writeLine(io.stderr, "data export requires --out PATH.zip");
          return 2;
        }
        return await (opts.data?.export ?? dataExport)(config, io, parsed.out);
      }
      if (!parsed.zip) {
        writeLine(io.stderr, "data import requires PATH.zip");
        return 2;
      }
      return await (opts.data?.import ?? dataImport)(config, io, parsed.zip, {
        yes: parsed.yes,
        isTTY: opts.isTTY ?? Boolean(process.stdin.isTTY),
      });
    }

    if (parsed.name === "email" && parsed.sub === "status") {
      return await emailStatus(config, io);
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
          return await emailTest(config, runtime.host, io, parsed.date, now);
        }
        return await emailVerifyStart(config, runtime.host, io);
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
      runtime.dispose?.();
    }
  } catch (e) {
    writeLine(io.stderr, (e as Error).message);
    return e instanceof CliConfigError ? 2 : 1;
  }

  writeLine(io.stderr, "internal: unhandled command");
  return 1;
}

const REMOVED_OVERRIDE_FLAGS = ["--config", "--vault-root", "--cache-dir"];

function parseCli(argv: string[]): CliCommand {
  for (const arg of argv) {
    const removedFlag = REMOVED_OVERRIDE_FLAGS.find(
      (flag) => arg === flag || arg.startsWith(`${flag}=`),
    );
    if (removedFlag) {
      throw new Error(
        `${removedFlag} is no longer supported; use ~/.config/arxiv-daily/config.toml (arxiv-daily init)`,
      );
    }
  }

  const rest: string[] = [];
  for (const arg of argv) {
    if (arg === "--help" || arg === "-h") return { name: "help" };
    rest.push(arg);
  }

  const [commandName, ...commandArgs] = rest;
  if (!commandName || commandName === "help") return { name: "help" };
  if (commandName === "init") return { name: "init" };
  if (commandName === "library") return { name: "library", args: commandArgs };
  if (commandName === "ui") {
    let port = 0;
    let open = true;
    let hasPort = false;
    let frameOrigin: string | undefined;
    for (let i = 0; i < commandArgs.length; i++) {
      const flag = commandArgs[i];
      if (flag === "--frame-origin" && frameOrigin === undefined) { frameOrigin = validateFrameOrigin(commandArgs[++i] || ""); continue; }
      if (flag === "--no-open") { open = false; continue; }
      if (flag === "--port" && !hasPort) {
        const value = commandArgs[++i];
        if (!value || !/^\d+$/.test(value) || Number(value) > 65535) throw new Error("ui --port requires an integer from 0 to 65535");
        port = Number(value);
        hasPort = true;
        continue;
      }
      throw new Error("ui accepts --port PORT, --no-open and --frame-origin ORIGIN only");
    }
    return { name: "ui", port, open, ...(frameOrigin ? { frameOrigin } : {}) };
  }
  if (commandName === "status") {
    if (commandArgs.length) throw new Error("status takes no arguments");
    return { name: "status" };
  }
  if (commandName === "papers") {
    for (let i = 0; i < commandArgs.length; i += 2) {
      if (!["--query", "--offset", "--limit"].includes(commandArgs[i] ?? "")) throw new Error("papers accepts --query, --offset, and --limit");
      optionValue(commandArgs.slice(i), commandArgs[i]!);
    }
    const offset = Number(optionValue(commandArgs, "--offset") ?? 0);
    const limit = Number(optionValue(commandArgs, "--limit") ?? 30);
    if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100) {
      throw new Error("papers requires offset >= 0 and limit 1..100");
    }
    return { name: "papers", query: optionValue(commandArgs, "--query") ?? "", offset, limit };
  }

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
