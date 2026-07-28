import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import type { CliRuntimeConfig } from "./config";
import { scheduleFireSlots } from "./config";
import type { CliIo } from "./main-types";

export const CRON_MARKER = "# arxiv-daily-managed";

export function buildCronLines(
  config: CliRuntimeConfig,
  binaryPath: string,
): string[] {
  const slots = scheduleFireSlots(config.scheduleIntent);
  const dow = config.scheduleIntent.weekdaysOnly ? "1-5" : "*";
  const lines: string[] = [];
  for (const slot of slots) {
    const [hh, mm] = slot.split(":");
    const hour = Number(hh);
    const minute = Number(mm);
    lines.push(
      `${minute} ${hour} * * ${dow}  ${binaryPath} run --today  ${CRON_MARKER}`,
    );
  }
  return lines;
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
  const managed = buildCronLines(config, binaryPath);
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

async function writeUserCrontab(body: string): Promise<void> {
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  const execFileAsync = promisify(execFile);
  const tmp = path.join(
    os.tmpdir(),
    `arxiv-daily-cron-${process.pid}-${Date.now()}`,
  );
  await fs.writeFile(tmp, body.endsWith("\n") ? body : `${body}\n`, "utf8");
  try {
    await execFileAsync("crontab", [tmp]);
  } finally {
    await fs.unlink(tmp).catch(() => undefined);
  }
}

function writeLine(stream: { write(chunk: string): unknown }, line: string): void {
  stream.write(`${line}\n`);
}
