import { spawn } from "node:child_process";
import * as path from "node:path";
import type { CliRuntimeConfig } from "./config";
import { scheduleFireSlots } from "./config";
import { resolveCliConfigPath } from "./config-path";
import type { CliIo } from "./main-types";

export const CRON_MARKER = "# arxiv-daily-managed";

/**
 * What cron needs to start the CLI: cron's PATH is minimal, so node is named explicitly, and its
 * environment is nearly empty, so a custom XDG_CONFIG_HOME travels on the command line.
 */
export interface CronRuntime {
  nodePath: string;
  logPath: string;
  /** XDG_CONFIG_HOME to set for the job; absent when the config is where cron's default looks. */
  configHome?: string;
}

export function defaultCronRuntime(config: CliRuntimeConfig): CronRuntime {
  const configHome = cronConfigHome(config.configPath);
  return {
    nodePath: process.execPath,
    logPath: path.join(path.dirname(config.configPath), "schedule.log"),
    ...(configHome ? { configHome } : {}),
  };
}

/**
 * The config path comes from $XDG_CONFIG_HOME (see config-path.ts), which a user usually exports from a
 * shell profile that cron never reads. Name it for the job unless the default location would find the
 * same file anyway. Native Windows has no cron, and a path outside the XDG layout cannot be named.
 */
function cronConfigHome(configPath: string): string | undefined {
  if (process.platform === "win32") return undefined;
  const absolute = path.resolve(configPath);
  if (absolute === resolveCliConfigPath({})) return undefined;
  const directory = path.dirname(absolute);
  if (path.basename(absolute) !== "config.toml" || path.basename(directory) !== "arxiv-daily") return undefined;
  return path.dirname(directory);
}

export function buildCronLines(
  config: CliRuntimeConfig,
  binaryPath: string,
  runtime: CronRuntime = defaultCronRuntime(config),
): string[] {
  if (!path.isAbsolute(binaryPath)) {
    throw new Error(`CLI script path must be absolute for cron: ${binaryPath}`);
  }
  // `node script`, not the script's shebang: `#!/usr/bin/env node` fails when node
  // is not on cron's PATH (nvm, fnm, volta, Homebrew on Apple Silicon).
  const environment = runtime.configHome ? `XDG_CONFIG_HOME=${cronExecutable(runtime.configHome)} ` : "";
  const executable = `${environment}${cronExecutable(runtime.nodePath)} ${cronExecutable(binaryPath)}`;
  const log = cronExecutable(runtime.logPath);
  const slots = scheduleFireSlots(config.scheduleIntent);
  const dow = config.scheduleIntent.weekdaysOnly ? "1-5" : "*";
  const lines: string[] = [];
  for (const slot of slots) {
    const [hh, mm] = slot.split(":");
    const hour = Number(hh);
    const minute = Number(mm);
    lines.push(
      `${minute} ${hour} * * ${dow}  ${executable} run --today >> ${log} 2>&1  ${CRON_MARKER}`,
    );
  }
  return lines;
}

function hasControlCharacter(value: string): boolean {
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code <= 0x1f || code === 0x7f) return true;
  }
  return false;
}

function cronExecutable(binaryPath: string): string {
  if (!binaryPath || hasControlCharacter(binaryPath)) {
    throw new Error("CLI executable path is empty or contains control characters");
  }
  // Cron removes the backslash before %, but preserves other shell escapes.
  // Escaping each character also preserves a literal backslash preceding %.
  return binaryPath.replace(/[^a-zA-Z0-9_./:-]/g, (character) => `\\${character}`);
}

export async function scheduleShow(
  config: CliRuntimeConfig,
  io: CliIo,
  binaryPath = resolveBinaryPath(),
): Promise<number> {
  const lines = buildCronLines(config, binaryPath);
  writeLine(io.stdout, `# from ${config.configPath}`);
  writeLine(
    io.stdout,
    `# schedule.enabled = ${config.scheduleIntent.enabled}`,
  );
  for (const line of lines) writeLine(io.stdout, line);
  if (!config.scheduleIntent.enabled) {
    writeLine(
      io.stderr,
      "note: schedule.enabled is false; install will refuse until you set enabled = true",
    );
  }
  return 0;
}

export async function scheduleInstall(
  config: CliRuntimeConfig,
  io: CliIo,
  opts: {
    binaryPath?: string;
    readCrontab?: () => Promise<string>;
    writeCrontab?: (body: string) => Promise<void>;
    platform?: NodeJS.Platform;
    runtime?: CronRuntime;
  } = {},
): Promise<number> {
  if (!config.scheduleIntent.enabled) {
    writeLine(
      io.stderr,
      "schedule.enabled is false; set enabled = true in config.toml then retry",
    );
    return 2;
  }
  const platform = opts.platform ?? process.platform;
  const binaryPath = opts.binaryPath ?? resolveBinaryPath();
  const runtime = opts.runtime ?? defaultCronRuntime(config);
  let managed: string[];
  try {
    managed = buildCronLines(config, binaryPath, runtime);
  } catch (error) {
    writeLine(io.stderr, `Invalid schedule: ${(error as Error).message}`);
    return 2;
  }
  if (platform === "win32" && !opts.writeCrontab) {
    writeLine(
      io.stderr,
      "schedule install uses crontab and is not supported on native Windows.",
    );
    writeLine(
      io.stderr,
      "On Windows: use WSL for the CLI + cron, or use the Obsidian plugin (in-app schedule while Obsidian is open).",
    );
    writeLine(io.stderr, "Cron lines that would be installed under Linux/WSL:");
    for (const line of managed) writeLine(io.stderr, line);
    return 2;
  }
  try {
    const current = opts.readCrontab
      ? await opts.readCrontab()
      : await readUserCrontab();
    const kept = current
      .split("\n")
      .filter((line) => !line.includes(CRON_MARKER));
    while (kept.length && kept[kept.length - 1] === "") kept.pop();
    const next = [...kept, ...managed, ""].join("\n");
    if (opts.writeCrontab) await opts.writeCrontab(next);
    else await writeUserCrontab(next);
    writeLine(io.stdout, `installed ${managed.length} cron line(s)`);
    for (const line of managed) writeLine(io.stdout, line);
    writeLine(io.stdout, `Output of scheduled runs is appended to ${runtime.logPath}`);
    if (runtime.configHome) {
      writeLine(io.stdout, `The schedule reads ${config.configPath} (XDG_CONFIG_HOME is set in each cron line); if you move the config, run: arxiv-daily schedule install`);
    }
    writeLine(
      io.stdout,
      `The schedule runs ${runtime.nodePath}; after upgrading or switching Node, run: arxiv-daily schedule install`,
    );
    return 0;
  } catch (e) {
    writeLine(io.stderr, `schedule install failed: ${(e as Error).message}`);
    writeLine(io.stderr, "Paste these lines into your crontab manually:");
    for (const line of managed) writeLine(io.stderr, line);
    return 1;
  }
}

export async function scheduleUninstall(
  config: CliRuntimeConfig,
  io: CliIo,
  opts: {
    readCrontab?: () => Promise<string>;
    writeCrontab?: (body: string) => Promise<void>;
    platform?: NodeJS.Platform;
  } = {},
): Promise<number> {
  void config;
  const platform = opts.platform ?? process.platform;
  if (platform === "win32" && !opts.writeCrontab) {
    writeLine(
      io.stderr,
      "schedule uninstall uses crontab and is not supported on native Windows. Use WSL, or remove jobs manually there; desktop users can rely on the Obsidian plugin scheduler.",
    );
    return 2;
  }
  try {
    const current = opts.readCrontab
      ? await opts.readCrontab()
      : await readUserCrontab();
    const kept = current
      .split("\n")
      .filter((line) => !line.includes(CRON_MARKER));
    const next = kept.join("\n").replace(/\n+$/, "") + "\n";
    if (opts.writeCrontab) await opts.writeCrontab(next);
    else await writeUserCrontab(next);
    writeLine(io.stdout, "removed managed arxiv-daily cron lines");
    return 0;
  } catch (e) {
    writeLine(io.stderr, `schedule uninstall failed: ${(e as Error).message}`);
    return 1;
  }
}

function resolveBinaryPath(): string {
  try {
    // Prefer the running entry when bundled
    if (process.argv[1] && !process.argv[1].includes("vitest")) {
      return path.resolve(process.argv[1]);
    }
  } catch {
    /* ignore */
  }
  return "arxiv-daily";
}

async function readUserCrontab(): Promise<string> {
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  const execFileAsync = promisify(execFile);
  try {
    const { stdout } = await execFileAsync("crontab", ["-l"], {
      encoding: "utf8",
    });
    return stdout;
  } catch (e) {
    const err = e as { stderr?: string; code?: number };
    // empty crontab often exits 1
    if (String(err.stderr ?? "").toLowerCase().includes("no crontab")) {
      return "";
    }
    throw e;
  }
}

/**
 * Install through `crontab -` on stdin: no temporary file, so there is no
 * predictable path for another local user to pre-plant a symlink at.
 */
export async function writeUserCrontab(body: string, command = "crontab"): Promise<void> {
  const text = body.endsWith("\n") ? body : `${body}\n`;
  await new Promise<void>((resolve, reject) => {
    const child = spawn(command, ["-"], { stdio: ["pipe", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.setEncoding("utf8").on("data", (chunk: string) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`crontab exited with code ${code}${stderr.trim() ? `: ${stderr.trim()}` : ""}`));
    });
    // A crontab that exits early closes the pipe; the exit code above reports why.
    child.stdin.on("error", () => undefined);
    child.stdin.end(text);
  });
}

function writeLine(stream: { write(chunk: string): unknown }, line: string): void {
  stream.write(`${line}\n`);
}
