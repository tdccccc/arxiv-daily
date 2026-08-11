import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { runCli, type CliCommandRuntime } from "../src/main";
import type { CliRuntimeConfig } from "../src/config";
import { DEFAULT_CLI_SCHEDULE } from "../src/config";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";

interface CompanionCliContract {
  schemaVersion: number;
  commands: Array<{ id: string; argv: string[] }>;
  removedCommands: Array<{ id: string; argv: string[] }>;
}

const companionCliContract = JSON.parse(
  readFileSync(new URL("../../../contracts/companion-cli-commands.json", import.meta.url), "utf8"),
) as CompanionCliContract;

function captureIo() {
  const stdout: string[] = [];
  const stderr: string[] = [];
  return {
    stdout,
    stderr,
    io: {
      stdout: { write: (chunk: string) => stdout.push(String(chunk)) },
      stderr: { write: (chunk: string) => stderr.push(String(chunk)) },
    },
  };
}

function testConfig(): CliRuntimeConfig {
  return {
    settings: {
      ...DEFAULT_SETTINGS,
      llm: { ...DEFAULT_SETTINGS.llm, apiKey: "key" },
      arxiv: {
        ...DEFAULT_SETTINGS.arxiv,
        timezone: "UTC",
        topics: [
          {
            id: "t1",
            name: "Topic",
            tag: "topic",
            description: "topic description",
            detail: false,
          },
        ],
      },
    },
    vaultRoot: "/vault",
    cacheDir: "/cache",
    linkStyle: "wikilink",
    configPath: "/home/u/.config/arxiv-daily/config.toml",
    scheduleIntent: { ...DEFAULT_CLI_SCHEDULE },
  };
}

function fakeRuntime(): CliCommandRuntime {
  return {
    pipeline: {
      runForDate: vi.fn(async () => ({
        kind: "completed" as const,
        papersWritten: 2,
      })),
    },
    manualFetch: {
      fetchAndSummarize: vi.fn(async () => ({
        kind: "done" as const,
        path: "papers/2606.12345.md",
      })),
    },
  };
}

describe("CLI main", () => {
  it("prints help without loading config", async () => {
    const io = captureIo();
    const loadConfig = vi.fn();
    const code = await runCli({
      argv: ["--help"],
      io: io.io,
      loadConfig: loadConfig as never,
    });
    expect(code).toBe(0);
    expect(io.stdout.join("")).toContain("arxiv-daily run --today");
    expect(loadConfig).not.toHaveBeenCalled();
  });

  it("rejects removed --config flag", async () => {
    const io = captureIo();
    const code = await runCli({
      argv: ["run", "--today", "--config", "x.toml"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => fakeRuntime(),
    });
    expect(code).toBe(2);
    expect(io.stderr.join("")).toContain("no longer supported");
  });

  it("runs pipeline for --date", async () => {
    const io = captureIo();
    const runtime = fakeRuntime();
    const code = await runCli({
      argv: ["run", "--date", "2026-06-13"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => runtime,
    });
    expect(code).toBe(0);
    expect(runtime.pipeline.runForDate).toHaveBeenCalledWith("2026-06-13");
  });

  it("runs --today using timezone", async () => {
    const io = captureIo();
    const runtime = fakeRuntime();
    const code = await runCli({
      argv: ["run", "--today"],
      io: io.io,
      now: () => new Date("2026-06-13T12:00:00Z"),
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => runtime,
    });
    expect(code).toBe(0);
    expect(runtime.pipeline.runForDate).toHaveBeenCalledWith("2026-06-13");
  });

  it("uses scheduler-backed run when available", async () => {
    const io = captureIo();
    const runtime = fakeRuntime();
    runtime.scheduler = {
      runForDateNow: vi.fn(async () => ({
        kind: "skipped" as const,
        reason: "already done",
      })),
    };
    const code = await runCli({
      argv: ["run", "--date", "2026-06-13"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => runtime,
    });
    expect(code).toBe(0);
    expect(runtime.scheduler.runForDateNow).toHaveBeenCalledWith("2026-06-13");
  });

  it("prints scheduler completion commit failures as existing transient errors", async () => {
    const io = captureIo();
    const runtime = fakeRuntime();
    runtime.scheduler = {
      runForDateNow: vi.fn(async () => ({
        kind: "failed_transient" as const,
        reason: "scheduler completion commit failed",
      })),
    };

    const code = await runCli({
      argv: ["run", "--date", "2026-06-13"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => runtime,
    });

    expect(code).toBe(1);
    expect(io.stdout.join("")).toBe("");
    expect(io.stderr.join("")).toContain(
      "run 2026-06-13: failed_transient (scheduler completion commit failed)",
    );
  });

  it("rejects run-pending", async () => {
    const io = captureIo();
    const code = await runCli({
      argv: ["run-pending"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => fakeRuntime(),
    });
    expect(code).toBe(2);
    expect(io.stderr.join("")).toContain("run-pending was removed");
  });

  it.each(companionCliContract.commands)(
    "accepts companion command contract argv for $id",
    async ({ id, argv }) => {
      expect(companionCliContract.schemaVersion).toBe(1);
      const io = captureIo();
      const runtime = fakeRuntime();
      const code = await runCli({
        argv,
        io: io.io,
        now: () => new Date("2026-06-13T12:00:00Z"),
        loadConfig: vi.fn(async () => testConfig()),
        buildRuntime: () => runtime,
      });
      expect(code, `${id}: ${io.stderr.join("")}`).toBe(0);
    },
  );

  it.each(companionCliContract.removedCommands)(
    "rejects removed companion argv for $id before config load",
    async ({ id, argv }) => {
      const io = captureIo();
      const loadConfig = vi.fn(async () => testConfig());
      const buildRuntime = vi.fn(() => fakeRuntime());
      const code = await runCli({
        argv,
        io: io.io,
        loadConfig,
        buildRuntime,
      });
      expect(code, `${id}: ${io.stderr.join("")}`).not.toBe(0);
      if (argv.some((arg) => /^(--config|--vault-root|--cache-dir)(?:=|$)/.test(arg))) {
        expect(code, id).toBe(2);
        expect(io.stderr.join(""), id).toContain("no longer supported");
        expect(loadConfig, id).not.toHaveBeenCalled();
        expect(buildRuntime, id).not.toHaveBeenCalled();
      }
    },
  );

  it("does not prefix-match longer names as removed flags", async () => {
    const io = captureIo();
    const loadConfig = vi.fn(async () => testConfig());
    const runtime = fakeRuntime();
    const code = await runCli({
      argv: [
        "run",
        "--today",
        "--config-extra=x",
        "--vault-rooted=/tmp/vault",
        "--cache-directory=/tmp/cache",
      ],
      io: io.io,
      now: () => new Date("2026-06-13T12:00:00Z"),
      loadConfig,
      buildRuntime: () => runtime,
    });
    expect(code, io.stderr.join("")).toBe(0);
    expect(loadConfig).toHaveBeenCalledOnce();
    expect(runtime.pipeline.runForDate).toHaveBeenCalledWith("2026-06-13");
  });

  it("runs --id for deep dive", async () => {
    const io = captureIo();
    const runtime = fakeRuntime();
    const code = await runCli({
      argv: ["run", "--id", "2606.12345", "--date", "2026-06-13"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => runtime,
    });
    expect(code).toBe(0);
    expect(runtime.manualFetch.fetchAndSummarize).toHaveBeenCalledWith(
      "2606.12345",
      "2026-06-13",
    );
    expect(io.stdout.join("")).toContain("run --id: wrote");
  });

  it("redacts secrets from errors", async () => {
    const io = captureIo();
    const secret = "sk-complete-secret-value";
    const runtime = fakeRuntime();
    runtime.pipeline.runForDate = vi.fn(async () => ({
      kind: "failed_transient" as const,
      reason: `provider echoed Bearer ${secret}`,
    }));
    const cfg = testConfig();
    cfg.settings.llm.apiKey = secret;
    const code = await runCli({
      argv: ["run", "--date", "2026-06-13"],
      io: io.io,
      loadConfig: vi.fn(async () => cfg),
      buildRuntime: () => runtime,
    });
    expect(code).toBe(1);
    expect(io.stderr.join("")).not.toContain(secret);
    expect(io.stderr.join("")).toContain("[REDACTED]");
  });

  it("schedule show prints cron lines", async () => {
    const io = captureIo();
    const cfg = testConfig();
    cfg.scheduleIntent = {
      enabled: true,
      on: "09:30",
      intervalHours: 0,
      until: "18:00",
      weekdaysOnly: true,
    };
    const code = await runCli({
      argv: ["schedule", "show"],
      io: io.io,
      loadConfig: vi.fn(async () => cfg),
      schedule: {
        show: async (c, i) => {
          i.stdout.write(`demo ${c.scheduleIntent.on}\n`);
          return 0;
        },
      },
    });
    expect(code).toBe(0);
    expect(io.stdout.join("")).toContain("09:30");
  });
});
