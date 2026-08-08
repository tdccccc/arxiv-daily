import { describe, expect, it, beforeAll, vi } from "vitest";
import {
  bindEnterToButton,
  isSupportedPaperIndexSchemaVersion,
  isValidCalendarDate,
  registerCommands,
} from "../src/commands";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { Modal, Notice } from "obsidian";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { FullTextRuntimeDiagnostics } from "../src/services/fulltext-runtime-diagnostics";

const commandsSource = readFileSync(resolve(process.cwd(), "src/commands.ts"), "utf8");

/**
 * Obsidian extends HTMLElement with createEl; the diagnostics modal's onOpen
 * uses it, so mirror the tiny surface here (same pattern as
 * personal-library-interest-profile-modal.test.ts).
 */
beforeAll(() => {
  type Options = { cls?: string; text?: string; attr?: Record<string, string> };
  const proto = HTMLElement.prototype as any;
  proto.createEl ??= function (tag: string, options: Options = {}) {
    const element = document.createElement(tag);
    if (options.cls) element.className = options.cls;
    if (options.text !== undefined) element.textContent = options.text;
    for (const [key, value] of Object.entries(options.attr ?? {})) element.setAttribute(key, value);
    this.appendChild(element);
    return element;
  };
});

/** Canned pass report for the full-text runtime diagnostics command. */
function passRuntimeDiagnostics(): FullTextRuntimeDiagnostics {
  return {
    library: { connected: true, scopeFingerprint: "scope-hex", paperCount: 5 },
    pdfJs: {
      status: "pass",
      loadPdfJsResolved: true,
      loaderReturnedLib: true,
      windowPdfJsLibPresent: true,
      windowPdfJsLibVersion: "4.2.189",
      smoke: { status: "pass", paperKey: "arXiv:1706.03762", pages: 15, chars: 12345 },
    },
    embedding: {
      status: "pass",
      modelId: "multilingual-e5-small-q8",
      dimension: 384,
      remoteHost: "https://huggingface.co/",
      loadMs: 1234,
    },
  };
}

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
    openPersonalLibraryDirectionReview: vi.fn(),
    diagnoseFullTextRuntime: vi.fn(async () => passRuntimeDiagnostics()),
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
  it.each([1, 2, 3, 4, 5])("accepts paper index schema %i", (schemaVersion) => {
    expect(isSupportedPaperIndexSchemaVersion(schemaVersion)).toBe(true);
  });

  it.each([0, 6, 99, -1, 2.5, "3", undefined])(
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

  it("registers and routes personal library direction review through the shared plugin entry", () => {
    const plugin = makePlugin();
    registerCommands(plugin as any);
    const command = vi.mocked(plugin.addCommand).mock.calls
      .map(([value]) => value)
      .find((value) => value.id === "review-personal-library-directions");

    expect(command?.name).toBe("Review personal library directions");
    command?.callback?.();
    expect(plugin.openPersonalLibraryDirectionReview).toHaveBeenCalledOnce();
    expect(plugin.addCommand).not.toHaveBeenCalledWith(
      expect.objectContaining({ id: expect.stringMatching(/generate.*library/i) }),
    );
  });

  it("contains synchronous direction review open failures without exposing details", () => {
    Notice.calls = [];
    const plugin = makePlugin();
    const hostile = new Error("/Users/alice/private/profile.json sha256:deadbeef");
    plugin.openPersonalLibraryDirectionReview.mockImplementation(() => { throw hostile; });
    registerCommands(plugin as any);
    const command = vi.mocked(plugin.addCommand).mock.calls
      .map(([value]) => value)
      .find((value) => value.id === "review-personal-library-directions");

    expect(() => command?.callback?.()).not.toThrow();
    expect(plugin.logger.error).toHaveBeenCalledWith(
      "commands: failed to open personal library direction review",
      hostile,
    );
    expect(Notice.calls.at(-1)?.message).toBe(
      "arXiv Daily: direction review could not be opened. Try again.",
    );
    expect(Notice.calls.at(-1)?.message).not.toContain("/Users/alice");
  });

  it("registers the full-text runtime diagnostics command and renders its report", async () => {
    Notice.calls = [];
    Modal.opened.length = 0;
    const plugin = makePlugin();
    const report = passRuntimeDiagnostics();
    plugin.diagnoseFullTextRuntime.mockResolvedValue(report);
    registerCommands(plugin as any);
    const command = vi.mocked(plugin.addCommand).mock.calls
      .map(([value]) => value)
      .find((value) => value.id === "diagnose-fulltext-runtime");

    expect(command?.name).toBe("Diagnose full-text runtime (pdf.js + embeddings)");
    command?.callback?.();
    for (let i = 0; i < 4; i++) await Promise.resolve();

    expect(plugin.diagnoseFullTextRuntime).toHaveBeenCalledOnce();
    expect(Notice.calls.at(-1)?.message).toContain("pdf.js PASS");
    expect(Notice.calls.at(-1)?.message).toContain("embeddings PASS");
    expect(Notice.calls.at(-1)?.message).toContain("smoke 15 pages / 12345 chars");
    const modal = Modal.opened.at(-1);
    expect(modal?.contentEl.textContent).toContain("full-text runtime diagnostics");
    const textarea = modal?.contentEl.querySelector("textarea");
    expect(textarea?.value).toContain("window.pdfjsLib: present (version 4.2.189)");
    expect(textarea?.value).toContain("smoke extraction: pass");
  });

  it("summarizes a failing runtime diagnostics run into FAIL notices", async () => {
    Notice.calls = [];
    const plugin = makePlugin();
    plugin.diagnoseFullTextRuntime.mockResolvedValue({
      library: { connected: false },
      pdfJs: {
        status: "fail",
        loadPdfJsResolved: false,
        loaderReturnedLib: false,
        windowPdfJsLibPresent: false,
        error: "loadPdfJs rejected",
      },
      embedding: {
        status: "fail",
        modelId: "multilingual-e5-small-q8",
        dimension: 384,
        error: "fetch failed",
      },
    });
    registerCommands(plugin as any);
    const command = vi.mocked(plugin.addCommand).mock.calls
      .map(([value]) => value)
      .find((value) => value.id === "diagnose-fulltext-runtime");

    command?.callback?.();
    for (let i = 0; i < 4; i++) await Promise.resolve();

    expect(Notice.calls.some((c) => c.message.includes("pdf.js FAIL"))).toBe(true);
    expect(Notice.calls.some((c) => c.message.includes("loadPdfJs rejected"))).toBe(true);
    expect(Notice.calls.some((c) => c.message.includes("embeddings FAIL"))).toBe(true);
    expect(Notice.calls.some((c) => c.message.includes("fetch failed"))).toBe(true);
  });

  it("notices and logs when the diagnostics probe itself rejects", async () => {
    Notice.calls = [];
    const plugin = makePlugin();
    const failure = new Error("probe crashed");
    plugin.diagnoseFullTextRuntime.mockRejectedValue(failure);
    registerCommands(plugin as any);
    const command = vi.mocked(plugin.addCommand).mock.calls
      .map(([value]) => value)
      .find((value) => value.id === "diagnose-fulltext-runtime");

    command?.callback?.();
    for (let i = 0; i < 4; i++) await Promise.resolve();

    expect(plugin.logger.error).toHaveBeenCalledWith(
      "commands: full-text runtime diagnostics failed",
      failure,
    );
    expect(Notice.calls.at(-1)?.message).toBe(
      "arXiv Daily: full-text runtime diagnostics failed: probe crashed",
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
