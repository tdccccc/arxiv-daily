import { describe, expect, it, vi } from "vitest";
import {
  bindEnterToButton,
  isSupportedPaperIndexSchemaVersion,
  isValidCalendarDate,
  registerCommands,
} from "../src/commands";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { Notice } from "obsidian";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const commandsSource = readFileSync(resolve(process.cwd(), "src/commands.ts"), "utf8");

function makePlugin() {
  const commands: Array<{ id: string; name: string; callback?: () => unknown }> = [];
  return {
    settings: DEFAULT_SETTINGS,
    app: {
      workspace: {
        openLinkText: vi.fn(),
      },
      vault: {
        adapter: {
          exists: vi.fn(async () => false),
        },
      },
    },
    manifest: { id: "arxiv-daily", version: "0.1.0" },
    scheduler: {
      runForDateNow: vi.fn(),
      runAllPending: vi.fn(),
      retryFailedInLookback: vi.fn(),
      forceRunForDate: vi.fn(),
      activeRuns: vi.fn(() => []),
      cancelCurrentRun: vi.fn(() => []),
    },
    stateStore: { clearAll: vi.fn(), snapshot: vi.fn(() => ({})) },
    runHistoryStore: { readLatest: vi.fn(async () => []) },
    progress: { setIdle: vi.fn() },
    logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    manualFetch: { fetchAndSummarize: vi.fn() },
    buildPaperIndex: vi.fn(() => ({
      paths: {
        papersJsonPath: "arxiv-daily/.index/papers.json",
        legacyPapersJsonPath: "arxiv-daily/papers.json",
      },
      get: vi.fn(),
      setStatus: vi.fn(),
      setPriority: vi.fn(),
    })),
    addCommand: vi.fn((command) => {
      commands.push(command);
    }),
    addRibbonIcon: vi.fn(() => ({ addClass: vi.fn() })),
  };
}

describe("isValidCalendarDate", () => {
  it("accepts real dates and rejects overflow dates", () => {
    expect(isValidCalendarDate("2024-02-29")).toBe(true);
    expect(isValidCalendarDate("2026-02-29")).toBe(false);
    expect(isValidCalendarDate("2026-04-31")).toBe(false);
    expect(isValidCalendarDate("2026-13-01")).toBe(false);
    expect(isValidCalendarDate("2026-7-01")).toBe(false);
  });
});

describe("isSupportedPaperIndexSchemaVersion", () => {
  it.each([1, 2, 3, 4])("accepts paper index schema %i", (schemaVersion) => {
    expect(isSupportedPaperIndexSchemaVersion(schemaVersion)).toBe(true);
  });

  it.each([0, 5, -1, 2.5, "3", undefined])(
    "rejects unsupported paper index schema %s",
    (schemaVersion) => {
      expect(isSupportedPaperIndexSchemaVersion(schemaVersion)).toBe(false);
    },
  );
});

describe("registerCommands", () => {
  it("registers the run history viewer command", () => {
    const plugin = makePlugin();

    registerCommands(plugin as any);

    expect(plugin.addCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "show-run-history",
        name: "Show run history",
      }),
    );
  });

  it("logs and notices rejected ribbon dashboard opens", async () => {
    Notice.calls = [];
    const plugin = makePlugin();
    const failure = new Error("leaf unavailable");
    plugin.app.workspace = {
      openLinkText: vi.fn(),
      getLeavesOfType: vi.fn(() => {
        throw failure;
      }),
    } as any;
    let ribbonCallback: (() => void) | undefined;
    plugin.addRibbonIcon.mockImplementation((_icon, _title, callback) => {
      ribbonCallback = callback;
      return { addClass: vi.fn() };
    });

    registerCommands(plugin as any);
    ribbonCallback?.();
    await Promise.resolve();
    await Promise.resolve();

    expect(plugin.logger.error).toHaveBeenCalledWith(
      "commands: failed to open dashboard from ribbon",
      failure,
    );
    expect(Notice.calls.at(-1)?.message).toContain(
      "failed to open dashboard from ribbon: leaf unavailable",
    );
  });

  it("leases paper-index and paper-note command writes through the output gate", () => {
    const markBody = commandsSource.match(
      /async function setPaperMark[\s\S]*?\n  async function createPaperNote/,
    )?.[0];
    const noteBody = commandsSource.match(
      /async function createPaperNote[\s\S]*?\n  function openArxivIdPicker/,
    )?.[0];

    expect(markBody).toContain("await plugin.withOutputOperation(");
    expect(markBody).toContain('mark === "saved" ? "paper-note" : "paper-index"');
    expect(noteBody).toContain("await plugin.withOutputOperation(");
    expect(noteBody).toContain('"paper-note"');
    expect(markBody?.indexOf("withOutputOperation")).toBeLessThan(
      markBody?.indexOf("buildPaperIndex") ?? -1,
    );
    expect(noteBody?.indexOf("withOutputOperation")).toBeLessThan(
      noteBody?.indexOf("buildPaperIndex") ?? -1,
    );
  });

  it("opens only done or verified already-existing manual summaries", () => {
    const body = commandsSource.match(
      /function openArxivIdPicker\(\)[\s\S]*?\n  async function openTodayDaily/,
    )?.[0];
    expect(body).toContain('result.kind === "done" || result.kind === "already_exists"');
    expect(body).toContain("refreshOpenDashboardViews(plugin)");
    expect(body).toContain('openLinkText(result.path, "", false)');
  });

  it("binds Enter in a single-field modal input to the submit button", () => {
    const input = document.createElement("input");
    const button = document.createElement("button");
    const click = vi.spyOn(button, "click");

    bindEnterToButton(input, button);
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );

    expect(click).toHaveBeenCalledTimes(1);
  });
});
