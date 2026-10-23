import { CliConfigError, loadCliConfig, type CliRuntimeConfig } from "./config";
import type { OperationRegistry, PipelineResult } from "@arxiv-daily/core";
import type { ManualFetchResult } from "@arxiv-daily/core";
import {
  deliverDailyEmailIfEnabled,
  describeTitleOnlyCandidates,
  resolveResendApiKey,
  validateFilterConfig,
  validateLlmConfig,
  formatDate,
  todayInTz,
  redactText,
  modernArxivResources,
} from "@arxiv-daily/core";
import type { HostAdapters } from "@arxiv-daily/core";
import { isCliCancelMessage, type CliIo, type WritableTextStream } from "./main-types";
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
import { diagnosticJson } from "./diagnostic-json";
import { runCliLibrary } from "./library-cmd";
import { runWorkbench } from "./workbench/launch";
import { validateFrameOrigin } from "./workbench/embedding";
import { isCalendarDate, parseOptions } from "./cli-args";

export type { CliIo, WritableTextStream } from "./main-types";

type CliRunResult = PipelineResult | { kind: "skipped"; reason: string };

export interface CliCommandRuntime {
  pipeline: {
    runForDate(date: string): Promise<PipelineResult>;
  };
  scheduler?: {
    runForDateNow(date: string, options?: { retryFailed?: boolean }): Promise<CliRunResult>;
    tick?: () => Promise<void>;
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
  /** Attached by the process entry point so a workbench cancel sent during startup is not lost. */
  workbenchCancel?: WorkbenchCancel;
}

type CliCommand =
  | { name: "help" }
  | { name: "init" }
  | { name: "status" }
  | { name: "ui"; port: number; open: boolean; frameOrigin?: string; exitOnStdinClose?: boolean }
  | { name: "papers"; query: string; offset: number; limit: number }
  | { name: "library"; args: string[] }
  | { name: "update"; checkOnly?: boolean; yes?: boolean }
  | { name: "run"; mode: "today" | "date" | "id" | "scheduled"; date?: string; id?: string }
  | { name: "email"; sub: "test" | "status" | "verify-start"; date?: string }
  | { name: "schedule"; sub: "show" | "install" | "uninstall" }
  | { name: "data"; sub: "export"; out?: string }
  | { name: "data"; sub: "import"; zip?: string; yes?: boolean };

const USAGE = `Usage:
  arxiv-daily init
  arxiv-daily status
  arxiv-daily ui [--port PORT] [--no-open] [--frame-origin ORIGIN] [--exit-on-stdin-close]
  arxiv-daily papers [--query TEXT] [--offset N] [--limit N]
  arxiv-daily library connect PATH
  arxiv-daily library status|prepare|scan|index|propose|directions|revoke
  arxiv-daily library authorize --fingerprint HASH
  arxiv-daily library confirm --candidate ID --proposal-revision N
  arxiv-daily library search --query TEXT [--mode hybrid|lexical|dense] [--limit N]
  arxiv-daily library review --input REQUEST.json
  arxiv-daily library maintain --other-clients-closed
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
  const rawIo: CliIo = opts.io ?? { stdout: process.stdout, stderr: process.stderr };
  let secrets: string[] = [];
  const io: CliIo = {
    onRunResult: rawIo.onRunResult,
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
        return await (opts.ui ?? runWorkbench)(undefined, io, { port: parsed.port, open: parsed.open, env, ...(parsed.frameOrigin ? { frameOrigin: parsed.frameOrigin } : {}), ...(parsed.exitOnStdinClose ? { exitOnStdinClose: true } : {}) });
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
      writeLine(rawIo.stdout, diagnosticJson(await inspectProduct(config), secrets));
      return 0;
    }
    if (parsed.name === "ui") {
      return await (opts.ui ?? runWorkbench)(config, io, { port: parsed.port, open: parsed.open, env, ...(parsed.frameOrigin ? { frameOrigin: parsed.frameOrigin } : {}), ...(parsed.exitOnStdinClose ? { exitOnStdinClose: true } : {}) });
    }
    if (parsed.name === "papers") {
      writeLine(rawIo.stdout, diagnosticJson(await inspectPapers(config, parsed.query, parsed.offset, parsed.limit), secrets));
      return 0;
    }
    if (parsed.name === "library") return await runCliLibrary(config, parsed.args, { ...io, stdout: rawIo.stdout });

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

    if (parsed.name === "run" && parsed.mode === "scheduled") {
      config = { ...config, settings: { ...config.settings, schedule: config.workbenchSchedule ?? config.settings.schedule } };
      if (!config.settings.schedule.enabled) return 0;
    }
    const runtime = await buildRuntime(config);
    const removeSignalHandlers = installSignalHandlers(runtime.operations, io, process, opts.workbenchCancel);
    try {
      if (opts.workbenchCancel?.requested) {
        writeLine(io.stderr, "arxiv-daily: cancelled before the run started");
        return 130;
      }
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
        if (parsed.mode === "scheduled") {
          if (!runtime.scheduler?.tick) throw new Error("Scheduled runs require the shared scheduler");
          await runtime.scheduler.tick();
          return 0;
        }
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
          ? await runtime.scheduler.runForDateNow(date, { retryFailed: true })
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
  if (commandName === "init") {
    parseOptions("init", commandArgs, {});
    return { name: "init" };
  }
  if (commandName === "library") return { name: "library", args: commandArgs };
  if (commandName === "ui") {
    let port = 0;
    let open = true;
    let hasPort = false;
    let frameOrigin: string | undefined;
    let exitOnStdinClose = false;
    for (let i = 0; i < commandArgs.length; i++) {
      const flag = commandArgs[i];
      if (flag === "--frame-origin" && frameOrigin === undefined) { frameOrigin = validateFrameOrigin(commandArgs[++i] || ""); continue; }
      if (flag === "--no-open") { open = false; continue; }
      if (flag === "--exit-on-stdin-close") { exitOnStdinClose = true; continue; }
      if (flag === "--port" && !hasPort) {
        const value = commandArgs[++i];
        if (!value || !/^\d+$/.test(value) || Number(value) > 65535) throw new Error("ui --port requires an integer from 0 to 65535");
        port = Number(value);
        hasPort = true;
        continue;
      }
      throw new Error("ui accepts --port PORT, --no-open, --frame-origin ORIGIN and --exit-on-stdin-close only");
    }
    return { name: "ui", port, open, ...(frameOrigin ? { frameOrigin } : {}), ...(exitOnStdinClose ? { exitOnStdinClose } : {}) };
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
    const { flags } = parseOptions("update", commandArgs, { flags: ["--check", "--yes", "-y"] });
    return { name: "update", checkOnly: flags.has("--check"), yes: flags.has("--yes") || flags.has("-y") };
  }

  if (commandName === "run") {
    const { flags, values } = parseOptions("run", commandArgs, { flags: ["--today", "--scheduled"], values: ["--date", "--id"] });
    if (flags.has("--scheduled")) {
      if (commandArgs.length !== 1) throw new Error("run --scheduled takes no other options");
      return { name: "run", mode: "scheduled" };
    }
    const today = flags.has("--today");
    const date = values["--date"];
    const id = values["--id"];
    if (date !== undefined) requireCalendarDate("run --date", date);
    // --id may also take --date for note dating
    if (id !== undefined) {
      if (today) throw new Error("run --id cannot be combined with --today");
      if (!modernArxivResources(id)) {
        throw new Error(`run --id requires an arXiv ID such as 2606.12345 (got ${JSON.stringify(id)})`);
      }
      return { name: "run", mode: "id", id, date };
    }
    if (today && date) throw new Error("run --today cannot be combined with --date");
    if (today) return { name: "run", mode: "today" };
    if (date) return { name: "run", mode: "date", date };
    throw new Error("run requires --today, --date YYYY-MM-DD, or --id ARXIV_ID");
  }

  if (commandName === "email" || commandName === "email-test") {
    const sub = commandName === "email-test" ? "test" : commandArgs[0];
    const rest = commandName === "email-test" ? commandArgs : commandArgs.slice(1);
    if (sub === "test") {
      const { values } = parseOptions("email test", rest, { values: ["--date"] });
      const date = values["--date"];
      if (date !== undefined) requireCalendarDate("email test --date", date);
      return { name: "email", sub: "test", date };
    }
    if (sub === "status" || sub === "verify-start") {
      parseOptions(`email ${sub}`, rest, {});
      return { name: "email", sub };
    }
    throw new Error('email requires subcommand: test | status | verify-start');
  }

  if (commandName === "schedule") {
    const sub = commandArgs[0];
    if (sub === "show" || sub === "install" || sub === "uninstall") {
      parseOptions(`schedule ${sub}`, commandArgs.slice(1), {});
      return { name: "schedule", sub };
    }
    throw new Error("schedule requires subcommand: show | install | uninstall");
  }

  if (commandName === "data") {
    const sub = commandArgs[0];
    if (sub === "export") {
      const { values } = parseOptions("data export", commandArgs.slice(1), { values: ["--out"] });
      return { name: "data", sub: "export", out: values["--out"] };
    }
    if (sub === "import") {
      const { flags, positionals } = parseOptions("data import", commandArgs.slice(1), { flags: ["--yes", "-y"], positionals: 1 });
      return { name: "data", sub: "import", zip: positionals[0], yes: flags.has("--yes") || flags.has("-y") };
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

function requireCalendarDate(option: string, value: string): void {
  if (!isCalendarDate(value)) throw new Error(`${option} requires a real calendar date in YYYY-MM-DD form (got ${JSON.stringify(value)})`);
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

/** Only a positive count is worth reporting; 0 keeps the event identical to before. */
function hasTitleOnlyCandidates(result: Extract<CliRunResult, { kind: "completed" }>): boolean {
  return describeTitleOnlyCandidates(result.titleOnlyCandidates) !== "";
}

function writeRunResult(io: CliIo, date: string, result: CliRunResult): number {
  io.onRunResult?.({ date, kind: result.kind,
    ...("outcome" in result ? { outcome: result.outcome } : {}),
    ...(result.kind === "completed" ? { papersWritten: result.papersWritten } : {}),
    ...(result.kind === "completed" && hasTitleOnlyCandidates(result) ? { titleOnlyCandidates: result.titleOnlyCandidates } : {}),
  });
  if (result.kind === "pending" && result.outcome === "awaiting_announcement") {
    writeLine(io.stdout, `run ${date}: awaiting_announcement (${result.reason})`);
    return 0;
  }
  if (result.kind === "completed") {
    const titleOnly = describeTitleOnlyCandidates(result.titleOnlyCandidates);
    writeLine(
      io.stdout,
      `run ${date}: completed (${result.papersWritten} papers written${titleOnly ? `; ${titleOnly}` : ""})${result.outcome ? ` [${result.outcome}]` : ""}`,
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

type SignalHost = Pick<NodeJS.Process, "on" | "off" | "exit">;

/**
 * The workbench cancels a run it spawned with an IPC message, not a second
 * signal (see CLI_CANCEL_MESSAGE). The process entry point attaches this before
 * its first await, so a cancel that arrives while the runtime is still being
 * built is remembered instead of dropped.
 */
export class WorkbenchCancel {
  requested = false;
  private subscriber: (() => void) | undefined;
  private readonly onMessage = (message: unknown) => {
    if (!isCliCancelMessage(message)) return;
    this.requested = true;
    this.subscriber?.();
  };

  constructor(private readonly host: Pick<NodeJS.Process, "on" | "off" | "connected"> = process) {
    // Only a process the workbench spawned has a channel. Its listener keeps
    // the process alive, so stop() must run once the command has finished.
    if (host.connected) host.on("message", this.onMessage);
  }

  subscribe(subscriber: (() => void) | undefined): void {
    this.subscriber = subscriber;
  }

  stop(): void {
    this.host.off("message", this.onMessage);
  }
}

export function installSignalHandlers(
  operations: OperationRegistry | undefined,
  io: CliIo,
  host: SignalHost = process,
  workbenchCancel?: WorkbenchCancel,
): () => void {
  if (!operations || !host.on) return () => {};
  const cancel = (reason: string, source: string) => {
    const active = operations.snapshot();
    operations.cancelAll(reason);
    writeLine(
      io.stderr,
      `arxiv-daily: ${source}; cancelling ${active.length} active task${active.length === 1 ? "" : "s"} and waiting`,
    );
  };
  let signalCount = 0;
  const handler = (signal: NodeJS.Signals) => {
    signalCount += 1;
    if (signalCount === 1) {
      cancel(`cancelled by ${signal}`, `${signal} received`);
      return;
    }
    host.exit(128 + (signal === "SIGINT" ? 2 : 15));
  };
  host.on("SIGINT", handler);
  host.on("SIGTERM", handler);
  // A workbench cancel does not count toward the forced exit above, so a
  // terminal Ctrl+C (which signals this process as well) still lets the
  // cancellation be written back.
  workbenchCancel?.subscribe(() => cancel("cancelled by user", "cancel requested by the workbench"));
  return () => {
    host.off("SIGINT", handler);
    host.off("SIGTERM", handler);
    workbenchCancel?.subscribe(undefined);
  };
}

async function defaultBuildRuntime(
  config: CliRuntimeConfig,
): Promise<CliCommandRuntime> {
  const { buildCliRuntime } = await import("./runtime");
  return buildCliRuntime(config);
}

if (typeof require !== "undefined" && require.main === module) {
  const workbenchCancel = new WorkbenchCancel();
  void runCli({ workbenchCancel, io: {
    stdout: process.stdout,
    stderr: process.stderr,
    onRunResult: event => {
      if (process.connected && process.send) process.send({ type: "arxiv-daily/run-result", event }, () => {});
    },
  } })
    .then((code) => {
      process.exitCode = code;
    })
    .catch((err) => {
      console.error(err);
      process.exit(1);
    })
    .finally(() => workbenchCancel.stop());
}
