import { spawn } from "node:child_process";
import {
  CLI_PACKAGE_NAME,
  compareSemver,
  getCliVersion,
} from "./version";
import type { CliIo } from "./main-types";

export interface UpdateOptions {
  /** Only print versions; never install. */
  checkOnly?: boolean;
  /** Install without interactive confirm. */
  yes?: boolean;
  isTTY?: boolean;
  /** Injected for tests. */
  fetchLatest?: () => Promise<string>;
  runInstall?: (spec: string) => Promise<{ code: number; output: string }>;
  confirm?: () => Promise<boolean>;
}

export async function runUpdate(
  io: CliIo,
  opts: UpdateOptions = {},
): Promise<number> {
  const current = getCliVersion();
  writeLine(io.stdout, `Current version: ${current}`);

  let latest: string;
  try {
    latest = opts.fetchLatest
      ? await opts.fetchLatest()
      : await fetchLatestFromNpm();
  } catch (e) {
    writeLine(
      io.stderr,
      `Could not check npm for updates: ${(e as Error).message}`,
    );
    writeLine(
      io.stderr,
      `Try manually: npm install -g ${CLI_PACKAGE_NAME}@latest`,
    );
    return 1;
  }

  writeLine(io.stdout, `Latest on npm:  ${latest}`);
  const cmp = compareSemver(current, latest);
  if (cmp >= 0) {
    writeLine(io.stdout, "Already up to date.");
    return 0;
  }

  writeLine(io.stdout, `Update available: ${current} → ${latest}`);
  if (opts.checkOnly) {
    writeLine(
      io.stdout,
      `Run: arxiv-daily update   or   npm install -g ${CLI_PACKAGE_NAME}@${latest}`,
    );
    return 0;
  }

  const tty = opts.isTTY ?? Boolean(process.stdin.isTTY);
  if (!opts.yes) {
    if (!tty && !opts.confirm) {
      writeLine(
        io.stderr,
        "Non-interactive update requires --yes (or use --check).",
      );
      writeLine(
        io.stdout,
        `Manual: npm install -g ${CLI_PACKAGE_NAME}@${latest}`,
      );
      return 2;
    }
    const ok = opts.confirm
      ? await opts.confirm()
      : await defaultConfirm(io, latest);
    if (!ok) {
      writeLine(io.stdout, "Update cancelled.");
      return 0;
    }
  }

  const spec = `${CLI_PACKAGE_NAME}@${latest}`;
  writeLine(io.stdout, `Installing ${spec} globally…`);
  const result = opts.runInstall
    ? await opts.runInstall(spec)
    : await npmInstallGlobal(spec);

  if (result.output.trim()) {
    writeLine(io.stdout, result.output.trimEnd());
  }
  if (result.code !== 0) {
    writeLine(io.stderr, `Update failed (exit ${result.code}).`);
    writeLine(
      io.stderr,
      `Try: npm install -g ${spec}`,
    );
    return 1;
  }

  writeLine(
    io.stdout,
    `Updated. Open a new shell if the command path is cached, then: arxiv-daily help`,
  );
  writeLine(
    io.stdout,
    `Your config at ~/.config/arxiv-daily/config.toml is kept.`,
  );
  return 0;
}

async function fetchLatestFromNpm(): Promise<string> {
  const url = `https://registry.npmjs.org/${CLI_PACKAGE_NAME}/latest`;
  const res = await fetch(url, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`npm registry HTTP ${res.status}`);
  }
  const data = (await res.json()) as { version?: string };
  if (!data.version || typeof data.version !== "string") {
    throw new Error("npm registry response missing version");
  }
  return data.version;
}

async function npmInstallGlobal(
  spec: string,
): Promise<{ code: number; output: string }> {
  return new Promise((resolve) => {
    const child = spawn("npm", ["install", "-g", spec], {
      env: process.env,
      shell: false,
    });
    let output = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      output += chunk.toString();
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      output += chunk.toString();
    });
    child.on("error", (err) => {
      resolve({ code: 1, output: err.message });
    });
    child.on("close", (code) => {
      resolve({ code: code ?? 1, output });
    });
  });
}

async function defaultConfirm(io: CliIo, latest: string): Promise<boolean> {
  writeLine(io.stdout, `Install arxiv-daily@${latest} globally now? [y/N]`);
  try {
    const readline = await import("node:readline/promises");
    const { stdin, stdout } = await import("node:process");
    const rl = readline.createInterface({ input: stdin, output: stdout });
    try {
      const ans = (await rl.question("")).trim().toLowerCase();
      return ans === "y" || ans === "yes";
    } finally {
      rl.close();
    }
  } catch {
    return false;
  }
}

function writeLine(stream: { write(chunk: string): unknown }, line: string): void {
  stream.write(`${line}\n`);
}
