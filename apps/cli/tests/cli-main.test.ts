import { describe, expect, it, vi } from "vitest";
import { runCli, type CliCommandRuntime } from "../src/main";
import type { CliRuntimeConfig } from "../src/config";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";

type TestSettingsOverrides = {
  llm?: Partial<CliRuntimeConfig["settings"]["llm"]>;
  arxiv?: Partial<CliRuntimeConfig["settings"]["arxiv"]>;
  output?: Partial<CliRuntimeConfig["settings"]["output"]>;
  schedule?: Partial<CliRuntimeConfig["settings"]["schedule"]>;
  advanced?: Partial<CliRuntimeConfig["settings"]["advanced"]>;
};

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

function testConfig(
  overrides: TestSettingsOverrides = {},
): CliRuntimeConfig {
  return {
    settings: {
      ...DEFAULT_SETTINGS,
      ...overrides,
      llm: {
        ...DEFAULT_SETTINGS.llm,
        apiKey: "key",
        ...(overrides.llm ?? {}),
      },
      arxiv: {
        ...DEFAULT_SETTINGS.arxiv,
        topics: [
          {
            id: "t1",
            name: "Topic",
            tag: "topic",
            description: "topic description",
            detail: false,
          },
        ],
        ...(overrides.arxiv ?? {}),
      },
      output: { ...DEFAULT_SETTINGS.output, ...(overrides.output ?? {}) },
      schedule: { ...DEFAULT_SETTINGS.schedule, ...(overrides.schedule ?? {}) },
      advanced: { ...DEFAULT_SETTINGS.advanced, ...(overrides.advanced ?? {}) },
    },
    vaultRoot: "/vault",
    cacheDir: "/cache",
    linkStyle: "wikilink",
    configPath: null,
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
      loadConfig: loadConfig as any,
    });

    expect(code).toBe(0);
    expect(io.stdout.join("")).toContain("arxiv-daily run --date");
    expect(loadConfig).not.toHaveBeenCalled();
  });

  it("runs the pipeline for an explicit date", async () => {
    const io = captureIo();
    const runtime = fakeRuntime();

    const code = await runCli({
      argv: ["run", "--date", "2026-06-13", "--config", "cfg.json"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => runtime,
    });

    expect(code).toBe(0);
    expect(runtime.pipeline.runForDate).toHaveBeenCalledWith("2026-06-13");
    expect(io.stdout.join("")).toContain("run 2026-06-13: completed");
  });

  it("uses scheduler-backed runs when available", async () => {
    const io = captureIo();
    const runtime = fakeRuntime();
    runtime.scheduler = {
      runForDateNow: vi.fn(async () => ({
        kind: "skipped" as const,
        reason: "already done",
      })),
      runAllPending: vi.fn(async () => []),
    };

    const code = await runCli({
      argv: ["run", "--date", "2026-06-13"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => runtime,
    });

    expect(code).toBe(0);
    expect(runtime.scheduler.runForDateNow).toHaveBeenCalledWith("2026-06-13");
    expect(runtime.pipeline.runForDate).not.toHaveBeenCalled();
    expect(io.stdout.join("")).toContain("skipped (already done)");
  });

  it("runs pending dates across the configured lookback window", async () => {
    const io = captureIo();
    const runtime = fakeRuntime();

    const code = await runCli({
      argv: ["run-pending"],
      io: io.io,
      now: () => new Date("2026-06-13T00:00:00Z"),
      loadConfig: vi.fn(async () =>
        testConfig({
          arxiv: { timezone: "UTC" },
        }),
      ),
      buildRuntime: () => runtime,
    });

    expect(code).toBe(0);
    expect(runtime.pipeline.runForDate).toHaveBeenCalledTimes(5);
    expect(runtime.pipeline.runForDate).toHaveBeenNthCalledWith(1, "2026-06-09");
    expect(runtime.pipeline.runForDate).toHaveBeenNthCalledWith(5, "2026-06-13");
  });

  it("summarizes one arXiv ID with an optional date", async () => {
    const io = captureIo();
    const runtime = fakeRuntime();

    const code = await runCli({
      argv: ["summarize", "--id", "2606.12345", "--date", "2026-06-13"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig()),
      buildRuntime: () => runtime,
    });

    expect(code).toBe(0);
    expect(runtime.manualFetch.fetchAndSummarize).toHaveBeenCalledWith(
      "2606.12345",
      "2026-06-13",
    );
    expect(io.stdout.join("")).toContain("summarize: wrote");
  });

  it("redacts the configured key from runtime errors and result presentation", async () => {
    const io = captureIo();
    const secret = "sk-complete-secret-value";
    const runtime = fakeRuntime();
    runtime.pipeline.runForDate = vi.fn(async () => ({
      kind: "failed_transient" as const,
      reason: `provider echoed Bearer ${secret}`,
    }));

    const code = await runCli({
      argv: ["run", "--date", "2026-06-13"],
      io: io.io,
      loadConfig: vi.fn(async () => testConfig({ llm: { apiKey: secret } })),
      buildRuntime: () => runtime,
    });

    expect(code).toBe(1);
    expect(io.stderr.join("")).not.toContain(secret);
    expect(io.stderr.join("")).toContain("[REDACTED]");
  });

  it("returns usage errors for invalid config", async () => {
    const io = captureIo();

    const code = await runCli({
      argv: ["run", "--date", "2026-06-13"],
      io: io.io,
      loadConfig: vi.fn(async () =>
        testConfig({
          llm: { ...DEFAULT_SETTINGS.llm, apiKey: "" },
        }),
      ),
      buildRuntime: () => fakeRuntime(),
    });

    expect(code).toBe(2);
    expect(io.stderr.join("")).toContain("Invalid config");
  });
});
