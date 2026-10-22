import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  RunCancellationService,
  RunCancelledError,
  RunLock,
  SchedulerService,
  StateStore,
  type PipelineResult,
  type StorageAdapter,
} from "@arxiv-daily/core";
import * as obsidianHost from "../src/hosts/obsidian";
import { chooseModal } from "../src/services/modal";
import { registerCommands } from "../src/commands";
import { withBusyGuard } from "../src/services/status-bar";
import ArxivDailyPlugin from "../main.ts";

vi.mock("../src/services/modal", () => ({ chooseModal: vi.fn() }));

const TODAY = "2026-10-21";
const RUN_STATE = "arxiv-daily/.index/run-state.json";

/** Obsidian's scoped element creation, which the progress panel uses on `body`. */
function installCreateEl(parent: HTMLElement): void {
  const create = function (
    this: HTMLElement,
    tag: string,
    options?: { cls?: string; attr?: Record<string, string> },
  ): HTMLElement {
    const child = this.ownerDocument.createElement(tag);
    if (options?.cls) child.className = options.cls;
    for (const [name, value] of Object.entries(options?.attr ?? {})) child.setAttribute(name, value);
    installCreateEl(child);
    this.append(child);
    return child;
  };
  (parent as any).createEl = create;
  (parent as any).createDiv = function (this: HTMLElement, options?: { cls?: string }): HTMLElement {
    return create.call(this, "div", options);
  };
}

function memoryStorage(files: Record<string, string>): StorageAdapter {
  const data = new Map(Object.entries(files));
  return {
    normalizePath: (path) => path.replace(/\\/g, "/"),
    readText: async (path) => {
      const value = data.get(path);
      if (value === undefined) throw new Error(`missing: ${path}`);
      return value;
    },
    writeText: async (path, content) => { data.set(path, content); },
    exists: async (path) => data.has(path),
    mkdir: async () => {},
    remove: async (path) => { data.delete(path); },
    rename: async (from, to) => { data.set(to, data.get(from)!); data.delete(from); },
    writeTextAtomic: async (path, content) => { data.set(path, content); },
    acquireLock: vi.fn(async () => null),
    list: vi.fn(async () => []),
  };
}

const completedRunState = (date: string) => JSON.stringify({
  schemaVersion: 1,
  runState: { [date]: { status: "completed", lastAttempt: 1, attempts: 1, papersWritten: 3 } },
});

/** Real `onload` wiring (status bar, operation subscription, scheduler); it stops when the settings tab is registered. */
async function loadPlugin(options: { enabled?: boolean; files?: Record<string, string> } = {}) {
  const plugin = new ArxivDailyPlugin();
  const statusEl = document.createElement("span");
  document.body.append(statusEl);
  const buildHost = vi.spyOn(obsidianHost, "buildObsidianHostAdapters")
    .mockReturnValue({ storage: memoryStorage(options.files ?? {}), http: {}, markupParser: {} } as never);
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.arxiv.timezone = "UTC";
  settings.schedule.enabled = options.enabled ?? false;
  settings.llm.apiKey = "configured";
  settings.arxiv.topics.push({
    id: "topic-1",
    name: "Language models",
    tag: "language-models",
    description: "Language model research",
    directions: [{ id: "d1", text: "Language model research", origin: "manual" }],
    detail: false,
  });
  Object.assign(plugin, {
    settings,
    app: {},
    manifest: { id: "arxiv-daily" },
    loadSettingsAndState: vi.fn(async () => []),
    cleanupCachesIfDue: vi.fn(),
    saveData: vi.fn(),
    addStatusBarItem: () => statusEl,
    addSettingTab: () => { throw new Error("stop once the status bar is wired"); },
  });
  await expect(plugin.onload()).rejects.toThrow("stop once the status bar is wired");
  return {
    plugin,
    settings,
    statusEl,
    panel: () => document.body.querySelector(".arxiv-daily-progress"),
    dispose: () => { plugin.onunload(); buildHost.mockRestore(); statusEl.remove(); },
  };
}

/** The scheduler as `onload` builds it, over the plugin's own registry and status bar, with a stubbed pipeline. */
function schedulerFor(
  loaded: Awaited<ReturnType<typeof loadPlugin>>,
  runForDate: (date: string, signal?: AbortSignal) => Promise<PipelineResult>,
) {
  const { plugin, settings } = loaded;
  return new SchedulerService({
    getSettings: () => settings,
    store: plugin.stateStore,
    lock: new RunLock(),
    logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(), notice: vi.fn() } as never,
    runForDate,
    progress: withBusyGuard(plugin.progress, () => plugin.operations.snapshot().length > 0),
    cancellation: new RunCancellationService(plugin.operations),
    now: () => new Date(`${TODAY}T12:00:00Z`),
  });
}

beforeEach(() => {
  installCreateEl(document.body);
  vi.mocked(chooseModal).mockReset();
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
  vi.setSystemTime(new Date(`${TODAY}T12:00:00Z`));
});

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = "";
});

describe("the status bar after a cancelled task", () => {
  it.each([
    ["personal-library-scan", "Personal library scan"],
    ["personal-library-fulltext-index", "Indexing library"],
    ["personal-library-direction-generation", "Direction generation"],
    ["pdf-download", "PDF download"],
    ["detail-summary", "Detail summary"],
    ["paper-note", "Paper note"],
  ] as const)("returns to its resting text once a cancelled %s task has unwound", async (kind, label) => {
    const { statusEl, panel, dispose, plugin } = await loadPlugin();
    try {
      const operation = plugin.operations.begin(kind, label, "key");

      plugin.operations.cancelAll();
      expect(statusEl.textContent).toBe("arXiv: Cancelling active tasks · 1 unwinding");
      expect(panel()?.classList.contains("is-hidden")).toBe(false);

      operation.finish();
      expect(statusEl.textContent).toBe("arXiv: disabled");
      vi.advanceTimersByTime(1_500);
      expect(panel()?.classList.contains("is-hidden")).toBe(true);
    } finally {
      dispose();
    }
  });

  it("rests on the idle text with the last completed date while the scheduler is on", async () => {
    const { statusEl, dispose, plugin } = await loadPlugin({
      enabled: true,
      files: { [RUN_STATE]: completedRunState("2026-10-20") },
    });
    try {
      const operation = plugin.operations.begin("personal-library-scan", "Personal library scan", "key");
      plugin.operations.cancelAll();
      expect(statusEl.textContent).toBe("arXiv: Cancelling active tasks · 1 unwinding");

      operation.finish();

      expect(statusEl.textContent).toBe("arXiv: idle · last 2026-10-20");
    } finally {
      dispose();
    }
  });

  it("waits for every cancelled task before it lets go of the notice", async () => {
    const { statusEl, dispose, plugin } = await loadPlugin();
    try {
      const first = plugin.operations.begin("pdf-download", "PDF download", "a");
      const second = plugin.operations.begin("personal-library-scan", "Personal library scan", "b");
      plugin.operations.cancelAll();
      expect(statusEl.textContent).toBe("arXiv: Cancelling active tasks · 2 unwinding");

      first.finish();
      expect(statusEl.textContent).toBe("arXiv: Cancelling active tasks · 1 unwinding");

      second.finish();
      expect(statusEl.textContent).toBe("arXiv: disabled");
    } finally {
      dispose();
    }
  });

  it("leaves a task started during the unwinding with its own text", async () => {
    const { statusEl, dispose, plugin } = await loadPlugin();
    try {
      const cancelled = plugin.operations.begin("personal-library-scan", "Personal library scan", "a");
      plugin.operations.cancelAll();
      const next = plugin.operations.begin("pdf-download", "PDF download", "b");
      plugin.progress.setTask("Indexing library titles and abstracts", "embedding");

      cancelled.finish();
      next.finish();

      expect(statusEl.textContent).toBe("arXiv: Indexing library titles and abstracts · embedding");
    } finally {
      dispose();
    }
  });

  it("leaves a finished task's own completion text alone when nothing was cancelled", async () => {
    const { statusEl, dispose, plugin } = await loadPlugin();
    try {
      const operation = plugin.operations.begin("personal-library-scan", "Personal library scan", "key");
      plugin.progress.setTask("Scanning personal library", "Identifying library papers");
      plugin.progress.setComplete("Personal library scan complete");
      operation.finish();

      expect(statusEl.textContent).toBe("Complete");
    } finally {
      dispose();
    }
  });

  it("keeps a cancelled daily run ending on the idle text", async () => {
    const loaded = await loadPlugin({ enabled: true });
    try {
      let started!: () => void;
      const running = new Promise<void>((resolve) => { started = resolve; });
      const scheduler = schedulerFor(loaded, (_date, signal) => new Promise((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(new RunCancelledError()), { once: true });
        started();
      }));

      const run = scheduler.runForDateNow(TODAY);
      await running;
      loaded.plugin.operations.cancelAll();
      expect(loaded.statusEl.textContent).toBe("arXiv: Cancelling active tasks · 1 unwinding");
      await run;

      expect(loaded.plugin.operations.snapshot()).toEqual([]);
      expect(loaded.statusEl.textContent).toBe("arXiv: idle");
    } finally {
      loaded.dispose();
    }
  });
});

describe("the status bar and the scheduler toggle", () => {
  it("stops saying disabled as soon as the scheduler is switched on without running today", async () => {
    const { plugin, settings, statusEl, dispose } = await loadPlugin({
      files: { [RUN_STATE]: completedRunState("2026-10-20") },
    });
    try {
      expect(statusEl.textContent).toBe("arXiv: disabled");
      vi.spyOn(plugin as any, "chooseScheduleEnableAction").mockResolvedValue("skip");

      await expect(plugin.setScheduleEnabled(true)).resolves.toBe(true);

      expect(settings.schedule.enabled).toBe(true);
      expect(statusEl.textContent).toBe("arXiv: idle · last 2026-10-20");
    } finally {
      dispose();
    }
  });

  it("says disabled again when the scheduler is switched off", async () => {
    const { plugin, statusEl, dispose } = await loadPlugin({ enabled: true });
    try {
      expect(statusEl.textContent).toBe("arXiv: idle");

      await expect(plugin.setScheduleEnabled(false)).resolves.toBe(true);

      expect(statusEl.textContent).toBe("arXiv: disabled");
    } finally {
      dispose();
    }
  });

  it("stays disabled after a manual run while the scheduler is off", async () => {
    const loaded = await loadPlugin();
    try {
      const scheduler = schedulerFor(loaded, async () => ({ kind: "completed", papersWritten: 3 }));

      await expect(scheduler.runForDateNow(TODAY)).resolves.toMatchObject({ kind: "completed" });
      expect(loaded.statusEl.textContent).toBe("arXiv: disabled");

      vi.advanceTimersByTime(4_000);
      expect(loaded.statusEl.textContent).toBe("arXiv: disabled");
      expect(loaded.panel()?.classList.contains("is-hidden")).toBe(true);
    } finally {
      loaded.dispose();
    }
  });

  it("rests on the idle text after a manual run while the scheduler is on", async () => {
    const loaded = await loadPlugin({ enabled: true });
    try {
      const scheduler = schedulerFor(loaded, async () => ({ kind: "completed", papersWritten: 3 }));

      await scheduler.runForDateNow(TODAY);
      vi.advanceTimersByTime(4_000);

      expect(loaded.statusEl.textContent).toBe(`arXiv: idle · last ${TODAY}`);
    } finally {
      loaded.dispose();
    }
  });
});

describe("the status bar after the run state is cleared", () => {
  it("drops the last-run date the cleared state no longer has", async () => {
    const { plugin, statusEl, dispose } = await loadPlugin({
      enabled: true,
      files: { [RUN_STATE]: completedRunState("2026-10-20") },
    });
    try {
      const commands: Array<{ id: string; callback?: () => unknown }> = [];
      Object.assign(plugin, {
        addCommand: (command: { id: string; callback?: () => unknown }) => { commands.push(command); },
        addRibbonIcon: () => ({ addClass: vi.fn() }),
      });
      registerCommands(plugin);
      vi.mocked(chooseModal).mockResolvedValue("clear");
      expect(statusEl.textContent).toBe("arXiv: idle · last 2026-10-20");

      await (commands.find((command) => command.id === "clear-run-state")!.callback!() as Promise<void>);
      await vi.waitFor(() => expect(statusEl.textContent).toBe("arXiv: idle"));
      expect(plugin.stateStore.snapshot()).toEqual({});
    } finally {
      dispose();
    }
  });

  it("keeps saying disabled while the scheduler is off", async () => {
    const { plugin, statusEl, dispose } = await loadPlugin({
      files: { [RUN_STATE]: completedRunState("2026-10-20") },
    });
    try {
      const commands: Array<{ id: string; callback?: () => unknown }> = [];
      Object.assign(plugin, {
        addCommand: (command: { id: string; callback?: () => unknown }) => { commands.push(command); },
        addRibbonIcon: () => ({ addClass: vi.fn() }),
      });
      registerCommands(plugin);
      vi.mocked(chooseModal).mockResolvedValue("clear");

      await (commands.find((command) => command.id === "clear-run-state")!.callback!() as Promise<void>);

      await vi.waitFor(() => expect(plugin.stateStore.snapshot()).toEqual({}));
      expect(statusEl.textContent).toBe("arXiv: disabled");
    } finally {
      dispose();
    }
  });
});

describe("the status bar after the run-state location changes", () => {
  it("drops the old last-run date when the new location has no completed day", async () => {
    const { plugin, statusEl, dispose } = await loadPlugin({
      enabled: true,
      files: { [RUN_STATE]: completedRunState("2026-10-20") },
    });
    try {
      expect(statusEl.textContent).toBe("arXiv: idle · last 2026-10-20");
      const empty = new StateStore(async () => ({ runState: {} }), async () => {});
      await empty.load();

      (plugin as any).installOutputStores({ stateStore: empty, runHistoryStore: plugin.runHistoryStore });

      expect(statusEl.textContent).toBe("arXiv: idle");
    } finally {
      dispose();
    }
  });
});
