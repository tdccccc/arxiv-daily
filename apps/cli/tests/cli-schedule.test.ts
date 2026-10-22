import { execFile } from "node:child_process";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { promisify } from "node:util";
import { describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";
import { resolveCliConfigPath } from "../src/config-path";
import { buildCronLines, defaultCronRuntime, scheduleInstall, writeUserCrontab, type CronRuntime } from "../src/schedule-cmd";

const execFileAsync = promisify(execFile);
const runtime: CronRuntime = { nodePath: "/opt/node/bin/node", logPath: "/home/u/.config/arxiv-daily/schedule.log" };

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
      const line = buildCronLines(config(), binary, { nodePath: "/bin/echo", logPath: path.join(dir, "cron.log") })[0]!;
      await execFileAsync("/bin/sh", ["-c", commandBeforeCronStdin(line)]);
      // `echo` stands in for node: it records exactly which script and arguments were passed.
      expect((await fs.readFile(path.join(dir, "cron.log"), "utf8")).trimEnd()).toBe(`${binary} run --today`);
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

  it("names the node binary, the absolute script and a log redirect", () => {
    expect(buildCronLines(config(), "/usr/local/bin/arxiv-daily", runtime)).toEqual([
      "30 9 * * 1-5  /opt/node/bin/node /usr/local/bin/arxiv-daily run --today >> /home/u/.config/arxiv-daily/schedule.log 2>&1  # arxiv-daily-managed",
    ]);
  });

  it("defaults to the running node and a log beside the config file", () => {
    const [line] = buildCronLines(config(), "/usr/local/bin/arxiv-daily");
    expect(line).toContain(`${process.execPath.replace(/[^a-zA-Z0-9_./:-]/g, c => `\\${c}`)} /usr/local/bin/arxiv-daily run --today >> /schedule.log 2>&1`);
  });

  it("escapes spaces in the node, script and log paths", () => {
    const [line] = buildCronLines(config(), "/opt/my apps/arxiv-daily", { nodePath: "/Users/a b/.nvm/node", logPath: "/Users/a b/log 1.txt" });
    expect(line).toContain("/Users/a\\ b/.nvm/node /opt/my\\ apps/arxiv-daily run --today >> /Users/a\\ b/log\\ 1.txt 2>&1");
  });

  it("rejects a relative script path because cron has no working directory", () => {
    expect(() => buildCronLines(config(), "arxiv-daily", runtime)).toThrow(/absolute/);
  });

  it("tells the user to reinstall after changing Node", async () => {
    const { io, stdout } = captureIo();
    expect(await scheduleInstall(config(), io, { binaryPath: "/usr/bin/arxiv-daily", runtime, readCrontab: async () => "", writeCrontab: async () => undefined })).toBe(0);
    const text = stdout.join("");
    expect(text).toContain("schedule.log");
    expect(text).toContain("schedule install");
  });

  it.skipIf(process.platform === "win32")("installs through crontab stdin without a temporary file", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-fake-crontab-"));
    try {
      const fake = path.join(dir, "crontab");
      await fs.writeFile(fake, `#!/bin/sh\nprintf '%s\\n' "$@" > "${dir}/args"\ncat > "${dir}/stdin"\n`, { mode: 0o700 });
      const before = (await fs.readdir(os.tmpdir())).filter(name => name.startsWith("arxiv-daily-cron-"));
      await writeUserCrontab("0 0 * * * job", fake);
      expect((await fs.readFile(path.join(dir, "args"), "utf8"))).toBe("-\n");
      expect(await fs.readFile(path.join(dir, "stdin"), "utf8")).toBe("0 0 * * * job\n");
      expect((await fs.readdir(os.tmpdir())).filter(name => name.startsWith("arxiv-daily-cron-"))).toEqual(before);
    } finally {
      await fs.rm(dir, { recursive: true, force: true });
    }
  });

  it.skipIf(process.platform === "win32")("reports crontab failures with its stderr", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-fake-crontab-"));
    try {
      const fake = path.join(dir, "crontab");
      await fs.writeFile(fake, "#!/bin/sh\necho 'bad entry' >&2\nexit 1\n", { mode: 0o700 });
      await expect(writeUserCrontab("x", fake)).rejects.toThrow(/exited with code 1: bad entry/);
    } finally {
      await fs.rm(dir, { recursive: true, force: true });
    }
  });

  it("rejects an inverted repeat window without touching existing cron jobs", async () => {
    const cfg = config();
    cfg.scheduleIntent = { ...cfg.scheduleIntent, on: "18:00", until: "09:00", intervalHours: 1 };
    const { io, stderr } = captureIo();
    const readCrontab = vi.fn().mockResolvedValue("0 9 * * * old-command # arxiv-daily-managed\n");
    const writeCrontab = vi.fn();
    expect(await scheduleInstall(cfg, io, { binaryPath: "/usr/bin/arxiv-daily", runtime, readCrontab, writeCrontab })).toBe(2);
    expect(stderr.join("")).toContain("schedule.until");
    expect(readCrontab).not.toHaveBeenCalled();
    expect(writeCrontab).not.toHaveBeenCalled();
  });

  it.each([
    { on: "09:30", until: "18:00", intervalHours: 4, count: 3 },
    { on: "09:30", until: "09:30", intervalHours: 1, count: 1 },
    { on: "18:00", until: "09:00", intervalHours: 0, count: 1 },
  ])("installs $count jobs for a valid schedule $on/$until/$intervalHours", async (schedule) => {
    const cfg = config();
    cfg.scheduleIntent = { ...cfg.scheduleIntent, ...schedule };
    const { io } = captureIo();
    const writeCrontab = vi.fn();
    expect(await scheduleInstall(cfg, io, {
      binaryPath: "/usr/bin/arxiv-daily", runtime,
      readCrontab: async () => "0 0 * * * keep-me\n",
      writeCrontab,
    })).toBe(0);
    const written = writeCrontab.mock.calls[0]![0] as string;
    expect(written).toContain("0 0 * * * keep-me");
    expect(written.match(/arxiv-daily-managed/g)).toHaveLength(schedule.count);
  });
});

// cron starts jobs with `/bin/sh -c` and a near-empty environment: the shell profile where a custom
// XDG_CONFIG_HOME is usually exported is never read, so the job would look for the config elsewhere.
describe("config location under cron", () => {
  const posix = it.skipIf(process.platform === "win32");
  const customConfig = (dir: string): CliRuntimeConfig => ({ ...config(), configPath: path.join(dir, "arxiv-daily", "config.toml") });

  posix("pins a custom XDG_CONFIG_HOME on every managed line", () => {
    const cfg = customConfig("/custom/xdg");
    cfg.scheduleIntent = { ...cfg.scheduleIntent, on: "09:30", until: "18:00", intervalHours: 4 };
    const lines = buildCronLines(cfg, "/usr/local/bin/arxiv-daily", { ...defaultCronRuntime(cfg), nodePath: "/opt/node/bin/node" });
    expect(lines).toHaveLength(3);
    for (const line of lines) {
      expect(line).toMatch(/^\d+ \d+ \* \* 1-5 {2}XDG_CONFIG_HOME=\/custom\/xdg \/opt\/node\/bin\/node \/usr\/local\/bin\/arxiv-daily run --today >> \/custom\/xdg\/arxiv-daily\/schedule\.log 2>&1 {2}# arxiv-daily-managed$/);
    }
  });

  posix("pins the location in the default runtime too, so show and install agree", () => {
    const cfg = customConfig("/custom/xdg");
    expect(buildCronLines(cfg, "/usr/local/bin/arxiv-daily")[0]).toContain("  XDG_CONFIG_HOME=/custom/xdg ");
    expect(defaultCronRuntime(cfg).configHome).toBe("/custom/xdg");
  });

  posix("writes a relative XDG_CONFIG_HOME as an absolute path, since cron has no working directory", () => {
    expect(defaultCronRuntime(customConfig("relative/xdg")).configHome).toBe(path.resolve("relative/xdg"));
  });

  posix("keeps the output unchanged when the config is in the default location", () => {
    const cfg = { ...config(), configPath: resolveCliConfigPath({}) };
    expect(defaultCronRuntime(cfg).configHome).toBeUndefined();
    const [line] = buildCronLines(cfg, "/usr/local/bin/arxiv-daily");
    expect(line).not.toContain("XDG_CONFIG_HOME");
    expect(line).toMatch(/^30 9 \* \* 1-5 {2}\/\S* \/usr\/local\/bin\/arxiv-daily run --today >> /);
  });

  posix("leaves a config path that does not follow the XDG layout alone", () => {
    expect(defaultCronRuntime(config()).configHome).toBeUndefined();
    expect(defaultCronRuntime({ ...config(), configPath: "/etc/arxiv/settings.toml" }).configHome).toBeUndefined();
  });

  posix("tells the user which config the installed schedule is bound to", async () => {
    const cfg = customConfig("/custom/xdg");
    const { io, stdout } = captureIo();
    const writeCrontab = vi.fn();
    const runtimeForConfig = { ...defaultCronRuntime(cfg), nodePath: "/opt/node/bin/node" };
    expect(await scheduleInstall(cfg, io, { binaryPath: "/usr/bin/arxiv-daily", runtime: runtimeForConfig, readCrontab: async () => "", writeCrontab })).toBe(0);
    expect(writeCrontab.mock.calls[0]![0]).toContain("XDG_CONFIG_HOME=/custom/xdg /opt/node/bin/node /usr/bin/arxiv-daily run --today");
    expect(stdout.join("")).toContain(cfg.configPath);
  });

  posix.each([
    "xdg with space", "xdg%percent", "xdg$HOME", "xdg'\"quote", "xdg\\back\\%slash", "配置 é", "xdg`printf substituted`", "xdg;echo bad",
  ])("hands the job exactly the XDG_CONFIG_HOME %s that found the config, in cron's environment", async (name) => {
    const scratch = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-cron-xdg-"));
    try {
      const cfg = customConfig(path.join(scratch, name));
      await fs.mkdir(path.dirname(cfg.configPath), { recursive: true });
      const binary = path.join(scratch, "arxiv-daily");
      // Stands in for the CLI: reports the XDG_CONFIG_HOME it was started with (`env` forwards its environment).
      await fs.writeFile(binary, '#!/bin/sh\nprintf "%s" "${XDG_CONFIG_HOME-unset}"\n', { mode: 0o700 });
      const [line] = buildCronLines(cfg, binary, { ...defaultCronRuntime(cfg), nodePath: "/usr/bin/env" });
      await execFileAsync("env", ["-i", `HOME=${scratch}`, "PATH=/usr/bin:/bin", "/bin/sh", "-c", commandBeforeCronStdin(line!)]);
      const seen = await fs.readFile(path.join(path.dirname(cfg.configPath), "schedule.log"), "utf8");
      expect(resolveCliConfigPath({ XDG_CONFIG_HOME: seen === "unset" ? undefined : seen })).toBe(cfg.configPath);
    } finally {
      await fs.rm(scratch, { recursive: true, force: true });
    }
  });

  posix("starts a job for a config in the default location without any XDG_CONFIG_HOME", async () => {
    const scratch = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-cron-home-"));
    vi.stubEnv("HOME", scratch);
    try {
      const cfg = { ...config(), configPath: resolveCliConfigPath({}) };
      await fs.mkdir(path.dirname(cfg.configPath), { recursive: true });
      const binary = path.join(scratch, "arxiv-daily");
      await fs.writeFile(binary, '#!/bin/sh\nprintf "%s" "${XDG_CONFIG_HOME-unset}"\n', { mode: 0o700 });
      const [line] = buildCronLines(cfg, binary, { ...defaultCronRuntime(cfg), nodePath: "/usr/bin/env" });
      await execFileAsync("env", ["-i", `HOME=${scratch}`, "PATH=/usr/bin:/bin", "/bin/sh", "-c", commandBeforeCronStdin(line!)]);
      expect(await fs.readFile(path.join(path.dirname(cfg.configPath), "schedule.log"), "utf8")).toBe("unset");
    } finally {
      vi.unstubAllEnvs();
      await fs.rm(scratch, { recursive: true, force: true });
    }
  });
});
