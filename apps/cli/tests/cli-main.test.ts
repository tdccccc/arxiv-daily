import { isCliRunEvent } from "../src/main-types";
import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { runCli, type CliCommandRuntime } from "../src/main";
import { CliConfigError } from "../src/config";
import * as emailCommands from "../src/email-cmd";
import { buildNodeHostAdapters } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "../src/config";
import { DEFAULT_CLI_SCHEDULE } from "../src/config";
import { DEFAULT_SETTINGS, OperationRegistry, type Topic } from "@arxiv-daily/core";

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
            directions: [{ id: "d1", text: "topic description", origin: "migrated" }],
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
  it.each([
    ["init"], ["update"], ["schedule", "show"], ["schedule", "install"],
    ["schedule", "uninstall"], ["data", "export", "--out", "/tmp/unused.zip"],
    ["data", "import", "/tmp/unused.zip"], ["email", "status"],
    ["email", "test"], ["email", "verify-start"],
  ])("handles a rejected async %s command inside the CLI boundary", async (...argv) => {
    const capture = captureIo();
    const cfg = testConfig();
    const secret = "configured-cli-secret-value";
    cfg.settings.llm.apiKey = secret;
    const message = argv[0] === "init" || argv[0] === "update"
      ? "early command failed" : `command failed with ${secret}`;
    const fail = vi.fn(async () => { throw new Error(message); });
    vi.spyOn(emailCommands, "emailStatus").mockImplementation(fail);
    vi.spyOn(emailCommands, "emailTest").mockImplementation(fail);
    vi.spyOn(emailCommands, "emailVerifyStart").mockImplementation(fail);
    const runtime = { ...fakeRuntime(), host: buildNodeHostAdapters({ rootDir: "/tmp/unused-cli-host" }) };
    const code = await runCli({
      argv, io: capture.io, loadConfig: async () => cfg, buildRuntime: () => runtime,
      init: fail, update: fail,
      schedule: { show: fail, install: fail, uninstall: fail },
      data: { export: fail, import: fail },
    });
    expect(code).toBe(1);
    expect(fail).toHaveBeenCalledOnce();
    expect(capture.stderr.join("")).toContain("command failed");
    expect(capture.stderr.join("")).not.toContain(secret);
  });

  it("classifies asynchronous configuration failures as exit code 2", async () => {
    const capture = captureIo();
    expect(await runCli({
      argv: ["schedule", "show"], io: capture.io, loadConfig: async () => testConfig(),
      schedule: { show: async () => { throw new CliConfigError("invalid schedule"); } },
    })).toBe(2);
    expect(capture.stderr.join("")).toContain("invalid schedule");
  });

  it("keeps signal handlers until an asynchronous email command settles", async () => {
    const capture = captureIo();
    const cfg = testConfig();
    cfg.settings.email.to = "test@example.com";
    const operations = new OperationRegistry();
    const host = buildNodeHostAdapters({ rootDir: "/tmp/unused-cli-host" });
    let resolveResponse!: (value: { status: number; headers: {}; bodyText: string }) => void;
    const response = new Promise<{ status: number; headers: {}; bodyText: string }>((resolve) => { resolveResponse = resolve; });
    const request = vi.spyOn(host.http, "request").mockReturnValue(response);
    const beforeInt = process.listenerCount("SIGINT");
    const beforeTerm = process.listenerCount("SIGTERM");
    const running = runCli({
      argv: ["email", "verify-start"], io: capture.io, loadConfig: async () => cfg,
      buildRuntime: () => ({ ...fakeRuntime(), host, operations }),
    });
    try {
      await vi.waitFor(() => expect(request).toHaveBeenCalledOnce());
      expect(process.listenerCount("SIGINT")).toBe(beforeInt + 1);
      expect(process.listenerCount("SIGTERM")).toBe(beforeTerm + 1);
    } finally {
      resolveResponse({ status: 200, headers: {}, bodyText: '{"ok":true}' });
      await running;
    }
    expect(process.listenerCount("SIGINT")).toBe(beforeInt);
    expect(process.listenerCount("SIGTERM")).toBe(beforeTerm);
  });

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

  it.each([
    [{ kind: "pending", reason: "not published", outcome: "awaiting_announcement" }, "awaiting_announcement"],
    [{ kind: "completed", papersWritten: 0, outcome: "no_updates" }, "no_updates"],
    [{ kind: "completed", papersWritten: 0, outcome: "no_matches" }, "no_matches"],
  ] as const)("publishes structured run outcomes without treating waiting as failure: %s", async (result, outcome) => {
    const capture = captureIo();
    const onRunResult = vi.fn();
    const runtime = fakeRuntime();
    runtime.scheduler = { runForDateNow: vi.fn(async () => result) };
    const code = await runCli({
      argv: ["run", "--date", "2026-10-02"],
      io: { ...capture.io, onRunResult },
      loadConfig: async () => testConfig(), buildRuntime: () => runtime,
    });
    expect(code).toBe(0);
    expect(capture.stderr).toEqual([]);
    expect(capture.stdout.join("")).toContain(outcome);
    expect(onRunResult).toHaveBeenCalledWith(expect.objectContaining({ date: "2026-10-02", kind: result.kind, outcome }));
  });

  it("mentions title-only candidates in the completed line and run event only when positive", async () => {
    const run = async (result: unknown) => {
      const capture = captureIo();
      const onRunResult = vi.fn();
      const runtime = fakeRuntime();
      runtime.scheduler = { runForDateNow: vi.fn(async () => result) } as never;
      const code = await runCli({
        argv: ["run", "--date", "2026-10-02"],
        io: { ...capture.io, onRunResult },
        loadConfig: async () => testConfig(), buildRuntime: () => runtime,
      });
      return { code, capture, onRunResult };
    };
    const positive = await run({ kind: "completed", papersWritten: 2, outcome: "papers_written", titleOnlyCandidates: { count: 3, total: 40 } });
    expect(positive.code).toBe(0);
    expect(positive.capture.stderr).toEqual([]);
    expect(positive.capture.stdout.join("")).toBe(
      "run 2026-10-02: completed (2 papers written; 3/40 candidates judged by title only) [papers_written]\n",
    );
    expect(positive.onRunResult).toHaveBeenCalledWith(expect.objectContaining({
      date: "2026-10-02", kind: "completed", papersWritten: 2, titleOnlyCandidates: { count: 3, total: 40 },
    }));
    const noMatches = await run({ kind: "completed", papersWritten: 0, outcome: "no_matches", titleOnlyCandidates: { count: 3, total: 40 } });
    expect(noMatches.code).toBe(0);
    expect(noMatches.capture.stdout.join("")).toBe(
      "run 2026-10-02: completed (0 papers written; 3/40 candidates judged by title only) [no_matches]\n",
    );
    for (const titleOnlyCandidates of [{ count: 0, total: 40 }, undefined]) {
      const quiet = await run({ kind: "completed", papersWritten: 2, outcome: "papers_written", ...(titleOnlyCandidates ? { titleOnlyCandidates } : {}) });
      expect(quiet.capture.stdout.join("")).toBe("run 2026-10-02: completed (2 papers written) [papers_written]\n");
      expect(quiet.onRunResult.mock.calls[0][0]).not.toHaveProperty("titleOnlyCandidates");
    }
  });

  it("validates titleOnlyCandidates on the run event passed through the workbench envelope", () => {
    const base = { date: "2026-10-02", kind: "completed", papersWritten: 2 };
    expect(isCliRunEvent(base)).toBe(true);
    expect(isCliRunEvent({ ...base, titleOnlyCandidates: { count: 3, total: 40 } })).toBe(true);
    for (const bad of [{ count: -1, total: 4 }, { count: 5, total: 4 }, { count: 1.5, total: 4 }, { count: "1", total: 4 }, "3/40", null]) {
      expect(isCliRunEvent({ ...base, titleOnlyCandidates: bad })).toBe(false);
    }
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
    expect(runtime.scheduler.runForDateNow).toHaveBeenCalledWith("2026-06-13", expect.objectContaining({ retryFailed: true }));
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

  it.each([
    ["a direction longer than the report can carry", (topic: Topic) => ({ ...topic, directions: [{ id: "d1", text: "x".repeat(1200), origin: "manual" as const }] }), 'Topic 1 "Topic" direction 1 is 1200 characters long; the limit is 1000, shorten it'],
    ["a tag longer than the report can carry", (topic: Topic) => ({ ...topic, tag: "t".repeat(130) }), 'Topic 1 "Topic" tag is 130 characters long; the limit is 120, shorten the topic name'],
    ["two topics with one name", (topic: Topic) => topic, 'Topics 1 and 2 are both named "Topic"; give each topic its own name'],
  ])("refuses a saved config with %s before building any runtime", async (_name, change, reason) => {
    const config = testConfig();
    const first = change(config.settings.arxiv.topics[0]!);
    config.settings.arxiv.topics = _name === "two topics with one name"
      ? [first, { ...first, id: "t2", tag: "topic-2" }]
      : [first];
    const io = captureIo();
    const buildRuntime = vi.fn(() => fakeRuntime());

    const code = await runCli({ argv: ["run", "--today"], io: io.io, loadConfig: vi.fn(async () => config), buildRuntime });

    expect(code).toBe(2);
    expect(io.stderr.join("")).toContain(`Invalid config:\n${reason}`);
    expect(buildRuntime).not.toHaveBeenCalled();
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
      argv: ["run", "--today", "--config-extra=x"],
      io: io.io,
      now: () => new Date("2026-06-13T12:00:00Z"),
      loadConfig,
      buildRuntime: () => runtime,
    });
    // Reported as an ordinary unknown option, not as the removed --config override.
    expect(code).toBe(2);
    expect(io.stderr.join("")).toContain("unknown option --config-extra=x");
    expect(io.stderr.join("")).not.toContain("no longer supported");
    expect(loadConfig).not.toHaveBeenCalled();
    expect(runtime.pipeline.runForDate).not.toHaveBeenCalled();
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

  describe("strict argument parsing", () => {
    async function run(argv: string[]) {
      const capture = captureIo();
      const loadConfig = vi.fn(async () => testConfig());
      const buildRuntime = vi.fn(() => fakeRuntime());
      const hooks = {
        init: vi.fn(async () => 0), update: vi.fn(async () => 0),
        schedule: { show: vi.fn(async () => 0), install: vi.fn(async () => 0), uninstall: vi.fn(async () => 0) },
        data: { export: vi.fn(async () => 0), import: vi.fn(async () => 0) },
      };
      const code = await runCli({ argv, io: capture.io, loadConfig, buildRuntime, ...hooks });
      return { code, capture, loadConfig, buildRuntime, hooks };
    }

    it.each([
      ["run", "--today", "--bogus"],
      ["run", "--today", "--dry-run"],
      ["run", "--today", "extra"],
      ["run", "--today", "--today"],
      ["run", "--date", "2026-06-13", "--date", "2026-06-14"],
      ["run", "--date", "garbage"],
      ["run", "--date", "2026-02-30"],
      ["run", "--date", "2026-13-01"],
      ["run", "--date", "2026-6-1"],
      ["run", "--id", "notanid"],
      ["run", "--id", "2606.12345", "--date", "2026-02-30"],
      ["run", "--id", "2606.12345", "--today"],
      ["run", "--today", "--date", "2026-06-13"],
      ["run", "--date"],
      ["run"],
      ["run", "--scheduled", "--today"],
      ["email", "test", "--date", "zzz"],
      ["email", "test", "extra"],
      ["email", "status", "--x"],
      ["email-test", "--bogus"],
      ["data", "import", "a.zip", "--yses"],
      ["data", "import", "a.zip", "b.zip"],
      ["data", "export", "--out", "a.zip", "--force"],
      ["data", "export", "--out", "a.zip", "--out", "b.zip"],
      ["schedule", "show", "extra"],
      ["schedule", "install", "--now"],
      ["update", "--chek"],
      ["update", "now"],
      ["init", "--force"],
    ])("rejects %j with a usage error and does no work", async (...argv) => {
      const { code, capture, loadConfig, buildRuntime, hooks } = await run(argv);
      expect(code).toBe(2);
      expect(capture.stderr.join("")).toContain("Usage:");
      expect(loadConfig).not.toHaveBeenCalled();
      expect(buildRuntime).not.toHaveBeenCalled();
      expect(hooks.init).not.toHaveBeenCalled();
      expect(hooks.update).not.toHaveBeenCalled();
      expect(hooks.data.import).not.toHaveBeenCalled();
      expect(hooks.data.export).not.toHaveBeenCalled();
      expect(hooks.schedule.install).not.toHaveBeenCalled();
    });

    it("names the bad date and the bad option in the message", async () => {
      expect((await run(["run", "--date", "2026-02-30"])).capture.stderr.join("")).toContain("2026-02-30");
      expect((await run(["run", "--today", "--bogus"])).capture.stderr.join("")).toContain("unknown option --bogus");
      expect((await run(["run", "--id", "notanid"])).capture.stderr.join("")).toContain("notanid");
    });

    it.each([
      [["run", "--today"], "buildRuntime"],
      [["run", "--date", "2026-06-13"], "buildRuntime"],
      [["run", "--date", "2024-02-29"], "buildRuntime"],
      [["run", "--id", "2606.12345"], "buildRuntime"],
      [["run", "--id", "2606.12345v2", "--date", "2026-06-13"], "buildRuntime"],
      [["run", "--id", "https://arxiv.org/abs/2606.12345"], "buildRuntime"],
      [["run", "--date", "2026-06-13", "--id", "2606.12345"], "buildRuntime"],
      [["data", "import", "a.zip"], "import"],
      [["data", "import", "a.zip", "--yes"], "import"],
      [["data", "import", "--yes", "a.zip"], "import"],
      [["data", "import", "-y", "a.zip"], "import"],
      [["data", "export", "--out", "a.zip"], "export"],
      [["update"], "update"],
      [["update", "--check"], "update"],
      [["update", "--yes"], "update"],
      [["update", "-y", "--check"], "update"],
      [["init"], "init"],
      [["schedule", "show"], "show"],
      [["schedule", "install"], "install"],
      [["schedule", "uninstall"], "uninstall"],
    ] as const)("still accepts documented form %j", async (argv, target) => {
      const { code, capture, buildRuntime, hooks } = await run([...argv]);
      expect(capture.stderr.join("")).not.toContain("Usage:");
      expect(code).toBe(0);
      const handler = target === "buildRuntime" ? buildRuntime
        : target === "import" ? hooks.data.import : target === "export" ? hooks.data.export
        : target === "update" ? hooks.update : target === "init" ? hooks.init : hooks.schedule[target];
      expect(handler).toHaveBeenCalledOnce();
    });

    it("passes the parsed yes flag and zip path to data import", async () => {
      const { hooks } = await run(["data", "import", "-y", "a.zip"]);
      expect(hooks.data.import).toHaveBeenCalledWith(expect.anything(), expect.anything(), "a.zip", expect.objectContaining({ yes: true }));
    });

    it("accepts the documented companion contract commands", async () => {
      for (const command of companionCliContract.commands) {
        const { capture } = await run(command.argv);
        expect(capture.stderr.join("")).not.toContain("Usage:");
      }
    });

    it("accepts the scheduled form the workbench spawns and email test with a date", async () => {
      expect((await run(["run", "--scheduled"])).capture.stderr.join("")).not.toContain("Usage:");
      expect((await run(["email", "test", "--date", "2026-06-13"])).capture.stderr.join("")).not.toContain("Usage:");
      expect((await run(["email-test", "--date", "2026-06-13"])).capture.stderr.join("")).not.toContain("Usage:");
    });
  });
});
