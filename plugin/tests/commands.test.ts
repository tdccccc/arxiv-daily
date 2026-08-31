import { describe, expect, it, beforeAll, vi } from "vitest";
import {
  bindEnterToButton,
  collectPaperIndexDiagnostics,
  DiagnosticsModal,
  isSupportedPaperIndexSchemaVersion,
  isValidCalendarDate,
  registerCommands,
} from "../src/commands";
import {
  DEFAULT_SETTINGS,
  PaperIndexStore,
  type StorageAdapter,
} from "@arxiv-daily/core";
import { Modal, Notice } from "obsidian";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { FullTextRuntimeDiagnostics } from "../src/services/fulltext-runtime-diagnostics";
import { buildSafePluginDiagnosticsReport } from "../src/services/paper-index-diagnostics";

const commandsSource = readFileSync(resolve(process.cwd(), "src/commands.ts"), "utf8");

/**
 * Obsidian extends HTMLElement with createEl; the diagnostics modal's onOpen
 * uses it, so mirror the tiny surface here (same pattern as
 * personal-library-interest-profile-modal.test.ts).
 */
const elementPrototype = HTMLElement.prototype as HTMLElement & {
  createEl?: (tag: string, options?: { text?: string; cls?: string }) => HTMLElement;
  setText?: (text: string) => void;
};
elementPrototype.createEl ??= function (tag, options = {}) {
  const element = this.ownerDocument.createElement(tag);
  if (options.text) element.textContent = options.text;
  if (options.cls) element.className = options.cls;
  this.appendChild(element);
  return element;
};
elementPrototype.setText ??= function (text) {
  this.textContent = text;
};
(elementPrototype as HTMLElement & {
  createDiv?: (options?: { text?: string; cls?: string }) => HTMLElement;
}).createDiv ??= function (options = {}) {
  return (this as unknown as {
    createEl: (tag: string, options?: { text?: string; cls?: string }) => HTMLElement;
  }).createEl("div", options);
};

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

function makeStorage(initialFiles: Record<string, string>) {
  const files = { ...initialFiles };
  const storage = {
    normalizePath: (path: string) => path.replace(/\\/g, "/"),
    async readText(path: string) {
      if (!(path in files)) throw new Error(`missing ${path}`);
      return files[path];
    },
    async writeText(path: string, content: string) {
      files[path] = content;
    },
    async exists(path: string) {
      return path in files;
    },
    async mkdir() {},
    async rename(from: string, to: string) {
      files[to] = files[from];
      delete files[from];
    },
    async remove(path: string) {
      delete files[path];
    },
  } satisfies StorageAdapter;
  return storage;
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
    getLibraryConnectionStatus: vi.fn(() => ({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: "2026-08-02T12:00:00.000Z",
    })),
    previewLibraryInventory: vi.fn(async () => ({
      eligible: [{ path: "papers/a.pdf", size: 10 }],
      ignored: [],
      folders: 1,
      truncated: false,
    })),
    scanPersonalLibrary: vi.fn(async () => makeCatalog()),
    reloadPersonalLibraryCatalog: vi.fn(async () => makeCatalog()),
  };
}

/** Minimal catalog shape the summary modal renders. */
function makeCatalog() {
  return {
    revision: 3,
    papers: {},
    lastScan: {
      ready: 4,
      papers: 4,
      unresolved: 1,
      unrelated: 0,
      failed: 0,
      truncated: false,
    },
  } as any;
}

function makeDiagnosticsProbeFailurePlugin(
  probe: "exists" | "read",
  privateMarker: string,
) {
  const plugin = makePlugin();
  const primaryPath = "arxiv-daily/.index/papers.json";
  const paperPath = "arxiv-daily/papers/2608.10002.md";
  const privateCauseMarker = `${privateMarker}_CAUSE`;
  const failure = new Error(privateMarker, {
    cause: new Error(privateCauseMarker),
  });
  plugin.buildPaperIndex = vi.fn(() => ({
    paths: { papersJsonPath: primaryPath },
    inspect: vi.fn(async () => ({
      inbox: { schemaVersion: 4, updatedAt: "", papers: {} },
      document: {
        schemaVersion: 4,
        papers: {
          "arxiv:2608.10002": {
            arxivId: "2608.10002",
            status: "saved",
            priority: "normal",
            seenDates: ["2026-08-10"],
            paperPath,
          },
        },
      },
      sourcePath: primaryPath,
      recoveredFromBackup: false,
    })),
  })) as any;
  plugin.app.vault.adapter = {
    exists: vi.fn(async () => {
      if (probe === "exists") throw failure;
      return true;
    }),
    read: vi.fn(async () => {
      if (probe === "read") throw failure;
      return '---\narxiv_id: "2608.10002"\n---\n';
    }),
  } as any;
  return { plugin, primaryPath, privateCauseMarker };
}

function loggedWarnings(plugin: ReturnType<typeof makePlugin>): string {
  return plugin.logger.warn.mock.calls
    .flat()
    .map((value) => {
      if (!(value instanceof Error)) return String(value);
      const cause = (value as Error & { cause?: unknown }).cause;
      return `${value.message} ${cause instanceof Error ? cause.message : String(cause ?? "")}`;
    })
    .join("\n");
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

describe("collectPaperIndexDiagnostics", () => {
  it("classifies corrupt index diagnostics without exposing private parser content", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const privateMarker = "PRIVATE_DIAGNOSTIC_MARKER_7F3B";
    const store = new PaperIndexStore(
      makeStorage({ [primaryPath]: privateMarker }),
      DEFAULT_SETTINGS.output,
    );
    const plugin = makePlugin();
    plugin.buildPaperIndex = vi.fn(() => store as any);

    const report = await buildSafePluginDiagnosticsReport(plugin as any);
    const loggerText = plugin.logger.warn.mock.calls
      .flat()
      .map((value) => {
        if (!(value instanceof Error)) return String(value);
        const cause = (value as Error & { cause?: unknown }).cause;
        return `${value.message} ${cause instanceof Error ? cause.message : String(cause ?? "")}`;
      })
      .join("\n");

    expect(report).toContain(`path: ${primaryPath}`);
    expect(report).toContain("error: paper_index_invalid");
    expect(report).not.toContain(privateMarker);
    expect(loggerText).toContain("paper_index_invalid");
    expect(loggerText).not.toContain(privateMarker);
    expect(plugin.buildPaperIndex).toHaveBeenCalledTimes(1);
  });

  it.each(["exists", "read"] as const)(
    "classifies adapter.%s probe failures without leaking message or cause",
    async (probe) => {
      const privateMarker = `PRIVATE_MARKER_DIAGNOSTICS_${probe.toUpperCase()}`;
      const { plugin, primaryPath, privateCauseMarker } =
        makeDiagnosticsProbeFailurePlugin(probe, privateMarker);

      const report = await buildSafePluginDiagnosticsReport(plugin as any);
      const loggerText = loggedWarnings(plugin);

      expect(report).toContain(`path: ${primaryPath}`);
      expect(report).toContain("error: paper_index_unavailable");
      expect(report).not.toContain(privateMarker);
      expect(report).not.toContain(privateCauseMarker);
      expect(loggerText).toContain("paper_index_unavailable");
      expect(loggerText).not.toContain(privateMarker);
      expect(loggerText).not.toContain(privateCauseMarker);
    },
  );

  it("uses the validated backup recovery result when the primary is corrupt", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const files: Record<string, string> = {
      [primaryPath]: "{corrupt",
      [`${primaryPath}.bak`]: JSON.stringify({
        schemaVersion: 4,
        updatedAt: "2026-08-10T00:00:00.000Z",
        papers: {
          "arxiv:2608.10001": {
            paperKey: "arxiv:2608.10001",
            source: "arxiv",
            externalId: "2608.10001",
            arxivId: "2608.10001",
            title: "Recovered diagnostic paper",
            status: "saved",
            priority: "high",
            seenDates: ["2026-08-10"],
          },
        },
      }),
    };
    const dirs = new Set<string>();
    const storage = {
      normalizePath(path: string) {
        return path.replace(/\\/g, "/");
      },
      async readText(path: string) {
        if (!(path in files)) throw new Error(`missing ${path}`);
        return files[path];
      },
      async writeText(path: string, content: string) {
        files[path] = content;
      },
      async exists(path: string) {
        return path in files || dirs.has(path);
      },
      async mkdir(path: string) {
        dirs.add(path);
      },
      async rename(from: string, to: string) {
        files[to] = files[from];
        delete files[from];
      },
      async remove(path: string) {
        delete files[path];
        dirs.delete(path);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);
    const plugin = makePlugin();
    plugin.buildPaperIndex = vi.fn(() => store as any);
    plugin.app.vault.adapter = {
      exists: storage.exists,
      read: storage.readText,
    } as any;

    await expect(collectPaperIndexDiagnostics(plugin as any)).resolves.toMatchObject({
      path: primaryPath,
      exists: true,
      sourcePath: `${primaryPath}.bak`,
      recoveredFromBackup: true,
      schemaVersion: 4,
      total: 1,
      statusCounts: { saved: 1 },
    });
  });
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

  describe("personal library maintenance commands (moved off the settings Manage menu)", () => {
    function findCommand(plugin: ReturnType<typeof makePlugin>, id: string) {
      return vi.mocked(plugin.addCommand).mock.calls
        .map(([value]) => value)
        .find((value) => value.id === id);
    }

    it("registers preview, scan and reload as distinct commands", () => {
      const plugin = makePlugin();
      registerCommands(plugin as any);

      expect(findCommand(plugin, "preview-personal-library-files")?.name).toBe(
        "Preview personal library files",
      );
      expect(findCommand(plugin, "scan-personal-library")?.name).toBe(
        "Scan personal library folder (rebuild catalog)",
      );
      expect(findCommand(plugin, "reload-personal-library-catalog")?.name).toBe(
        "Reload personal library catalog from disk",
      );
    });

    it("routes preview to the plugin inventory preview and shows the modal", async () => {
      Notice.calls = [];
      Modal.opened.length = 0;
      const plugin = makePlugin();
      registerCommands(plugin as any);

      findCommand(plugin, "preview-personal-library-files")?.callback?.();
      for (let i = 0; i < 6; i++) await Promise.resolve();

      expect(plugin.previewLibraryInventory).toHaveBeenCalledOnce();
      expect(Modal.opened.at(-1)?.titleEl.textContent).toBe(
        "Personal library inventory",
      );
    });

    it("routes scan to a folder rescan and reload to a catalog reload", async () => {
      Modal.opened.length = 0;
      const plugin = makePlugin();
      registerCommands(plugin as any);

      findCommand(plugin, "scan-personal-library")?.callback?.();
      for (let i = 0; i < 6; i++) await Promise.resolve();
      expect(plugin.scanPersonalLibrary).toHaveBeenCalledOnce();
      expect(plugin.reloadPersonalLibraryCatalog).not.toHaveBeenCalled();

      findCommand(plugin, "reload-personal-library-catalog")?.callback?.();
      for (let i = 0; i < 6; i++) await Promise.resolve();
      expect(plugin.reloadPersonalLibraryCatalog).toHaveBeenCalledOnce();
      expect(plugin.scanPersonalLibrary).toHaveBeenCalledOnce();
      expect(Modal.opened.at(-1)?.titleEl.textContent).toBe(
        "Personal library catalog",
      );
    });

    it("asks for a folder instead of throwing when no library is connected", async () => {
      Notice.calls = [];
      const plugin = makePlugin();
      plugin.getLibraryConnectionStatus.mockReturnValue({ kind: "disconnected" } as any);
      registerCommands(plugin as any);

      for (const id of [
        "preview-personal-library-files",
        "scan-personal-library",
        "reload-personal-library-catalog",
      ]) {
        Notice.calls = [];
        expect(() => findCommand(plugin, id)?.callback?.()).not.toThrow();
        for (let i = 0; i < 6; i++) await Promise.resolve();
        expect(Notice.calls.at(-1)?.message).toBe(
          "arXiv Daily: choose a personal library folder in settings first.",
        );
      }
      expect(plugin.previewLibraryInventory).not.toHaveBeenCalled();
      expect(plugin.scanPersonalLibrary).not.toHaveBeenCalled();
      expect(plugin.reloadPersonalLibraryCatalog).not.toHaveBeenCalled();
    });

    it("notices and logs a failed scan without throwing", async () => {
      Notice.calls = [];
      const plugin = makePlugin();
      const failure = new Error("folder identity changed");
      plugin.scanPersonalLibrary.mockRejectedValue(failure);
      registerCommands(plugin as any);

      expect(() => findCommand(plugin, "scan-personal-library")?.callback?.()).not.toThrow();
      for (let i = 0; i < 6; i++) await Promise.resolve();

      expect(plugin.logger.error).toHaveBeenCalledWith(
        "commands: personal library scan failed",
        failure,
      );
      expect(Notice.calls.at(-1)?.message).toBe(
        "arXiv Daily: personal library scan failed: folder identity changed",
      );
    });
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

  it("renders corrupt-index diagnostics without rebuilding or leaking parser content", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const privateMarker = "PRIVATE_MODAL_INDEX_MARKER_4D8E";
    const store = new PaperIndexStore(
      makeStorage({ [primaryPath]: privateMarker }),
      DEFAULT_SETTINGS.output,
    );
    const plugin = makePlugin();
    plugin.buildPaperIndex = vi.fn(() => store as any);
    const modal = new DiagnosticsModal(plugin.app as any, plugin as any);

    modal.onOpen();
    const textarea = modal.contentEl.querySelector("textarea") as HTMLTextAreaElement;
    await vi.waitFor(() => {
      expect(textarea.value).not.toBe("Loading diagnostics…");
    });

    const loggerText = plugin.logger.warn.mock.calls.flat().join("\n");
    expect(textarea.value).toContain(`path: ${primaryPath}`);
    expect(textarea.value).toContain("error: paper_index_invalid");
    expect(textarea.value).not.toContain(privateMarker);
    expect(loggerText).toContain("paper_index_invalid");
    expect(loggerText).not.toContain(privateMarker);
    expect(plugin.buildPaperIndex).toHaveBeenCalledTimes(1);
  });

  it("renders adapter probe failures in the copyable modal report without leaking details", async () => {
    const privateMarker = "PRIVATE_MARKER_MODAL_PROBE";
    const { plugin, primaryPath, privateCauseMarker } =
      makeDiagnosticsProbeFailurePlugin("exists", privateMarker);
    const modal = new DiagnosticsModal(plugin.app as any, plugin as any);

    modal.onOpen();
    const textarea = modal.contentEl.querySelector("textarea") as HTMLTextAreaElement;
    await vi.waitFor(() => {
      expect(textarea.value).not.toBe("Loading diagnostics…");
    });

    const loggerText = loggedWarnings(plugin);
    expect(textarea.value).toContain(`path: ${primaryPath}`);
    expect(textarea.value).toContain("error: paper_index_unavailable");
    expect(textarea.value).not.toContain(privateMarker);
    expect(textarea.value).not.toContain(privateCauseMarker);
    expect(loggerText).toContain("paper_index_unavailable");
    expect(loggerText).not.toContain(privateMarker);
    expect(loggerText).not.toContain(privateCauseMarker);
  });

  it("does not repeat colliding output path derivation after store construction fails", async () => {
    const plugin = makePlugin();
    let dailyDirDerivations = 0;
    let papersDirDerivations = 0;
    const output = {
      get dailyDir() {
        if (new Error().stack?.includes("derivePaperInboxPaths")) {
          dailyDirDerivations += 1;
        }
        return "arxiv/collision";
      },
      get papersDir() {
        if (new Error().stack?.includes("derivePaperInboxPaths")) {
          papersDirDerivations += 1;
        }
        return "arxiv/collision";
      },
    };
    plugin.settings = { ...DEFAULT_SETTINGS, output };
    plugin.buildPaperIndex = vi.fn(() => new PaperIndexStore({
      normalizePath: (path: string) => path,
    } as StorageAdapter, output));
    const modal = new DiagnosticsModal(plugin.app as any, plugin as any);

    modal.onOpen();
    await Promise.resolve();
    await Promise.resolve();

    const textarea = modal.contentEl.querySelector("textarea") as HTMLTextAreaElement;
    expect(textarea.value).toContain("path: unavailable");
    expect(textarea.value).toContain("error: paper_index_configuration_invalid");
    expect(plugin.buildPaperIndex).toHaveBeenCalledTimes(1);
    expect(dailyDirDerivations).toBe(1);
    expect(papersDirDerivations).toBe(1);
  });

  it("uses the constructed store path when paper-index inspection fails", async () => {
    const plugin = makePlugin();
    const failure = new Error("inspection unavailable");
    plugin.buildPaperIndex = vi.fn(() => ({
      paths: { papersJsonPath: "custom/.index/papers.json" },
      inspect: vi.fn(async () => { throw failure; }),
    }));
    const modal = new DiagnosticsModal(plugin.app as any, plugin as any);

    modal.onOpen();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    const textarea = modal.contentEl.querySelector("textarea") as HTMLTextAreaElement;
    expect(textarea.value).toContain("path: custom/.index/papers.json");
    expect(textarea.value).toContain("error: paper_index_unavailable");
    expect(plugin.buildPaperIndex).toHaveBeenCalledTimes(1);
  });

  it("show-diagnostics command replaces loading text after collector failure", async () => {
    const plugin = makePlugin();
    plugin.buildPaperIndex = vi.fn(() => {
      throw new Error("paper index unavailable");
    });
    let opened: Modal | undefined;
    const open = vi.spyOn(Modal.prototype, "open").mockImplementation(function () {
      opened = this;
      this.onOpen();
    });
    registerCommands(plugin as any);
    const command = plugin.addCommand.mock.calls
      .map(([value]) => value)
      .find((value) => value.id === "show-diagnostics");

    command?.callback?.();
    await Promise.resolve();
    await Promise.resolve();

    const textarea = opened?.contentEl.querySelector("textarea") as HTMLTextAreaElement;
    expect(open).toHaveBeenCalledTimes(1);
    expect(textarea.value).toContain("error: paper_index_unavailable");
    expect(textarea.value).not.toContain("Loading diagnostics");
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
