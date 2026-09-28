import { execFile } from "node:child_process";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { promisify } from "node:util";
import { describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";
import { buildCronLines, scheduleInstall } from "../src/schedule-cmd";

const execFileAsync = promisify(execFile);

function config(): CliRuntimeConfig {
  return {
    settings: structuredClone(DEFAULT_SETTINGS),
    vaultRoot: "/vault", cacheDir: "/cache", configPath: "/config.toml",
    linkStyle: "wikilink",
    scheduleIntent: { ...DEFAULT_CLI_SCHEDULE, enabled: true },
  };
}

function captureIo() {
  const stdout: string[] = [];
  const stderr: string[] = [];
  return {
    stdout, stderr,
    io: {
      stdout: { write: (s: string) => stdout.push(s) },
      stderr: { write: (s: string) => stderr.push(s) },
    },
  };
}

// Cron processes percent escapes before invoking the shell; shell quotes do not
// protect an unescaped percent from becoming the command/stdin separator.
function commandBeforeCronStdin(line: string): string {
  const source = line.replace(/^(?:\S+\s+){5}/, "").trimStart();
  let command = "";
  let escaped = false;
  for (const ch of source) {
    if (escaped) {
      if (ch !== "%") command += "\\";
      command += ch;
      escaped = false;
    } else if (ch === "\\") {
      escaped = true;
    } else if (ch === "%") {
      throw new Error("unescaped cron percent starts stdin");
    } else {
      command += ch;
    }
  }
  return command + (escaped ? "\\" : "");
}

describe("CLI schedule command", () => {
  it.skipIf(process.platform === "win32").each([
    "arxiv daily", "arxiv'daily", 'arxiv"daily', "arxiv$daily",
    "arxiv`printf substituted`", "arxiv\\daily", "arxiv%daily", "arxiv\\%daily",
  ])("runs the exact executable path %s through cron and a POSIX shell", async (name) => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-cron-"));
    try {
      const binary = path.join(dir, name);
      await fs.writeFile(binary, '#!/bin/sh\nprintf "%s\\n" "$0" "$@"\n', { mode: 0o700 });
      const line = buildCronLines(config(), binary)[0]!;
      const { stdout } = await execFileAsync("/bin/sh", ["-c", commandBeforeCronStdin(line)]);
      expect(stdout.trimEnd().split("\n")).toEqual([binary, "run", "--today"]);
    } finally {
      await fs.rm(dir, { recursive: true, force: true });
    }
  });

  it.each(["/tmp/bad\npath", "/tmp/bad\rpath", "/tmp/bad\u0000path"])(
    "rejects an unrepresentable executable path before reading or writing crontab",
    async (binaryPath) => {
      const { io, stderr } = captureIo();
      const readCrontab = vi.fn().mockResolvedValue("existing task\n");
      const writeCrontab = vi.fn();
      expect(await scheduleInstall(config(), io, { binaryPath, readCrontab, writeCrontab })).toBe(2);
      expect(stderr.join("")).toContain("path");
      expect(readCrontab).not.toHaveBeenCalled();
      expect(writeCrontab).not.toHaveBeenCalled();
    },
  );

  it("preserves the ordinary executable command", () => {
    expect(buildCronLines(config(), "/usr/local/bin/arxiv-daily")).toEqual([
      "30 9 * * 1-5  /usr/local/bin/arxiv-daily run --today  # arxiv-daily-managed",
    ]);
  });
});
